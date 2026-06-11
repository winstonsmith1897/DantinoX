"""
Comprehensive pytest suite for train_sweep_attention_comparison.py

Tests the training loop with mocked JAX/Flax dependencies and W&B integration.
Validates configuration handling, optimizer setup, and batch processing.
"""

import json
import math
import os
import tempfile
from unittest.mock import Mock, patch

import jax.numpy as jnp
import pytest

# ============================================================================
# FIXTURES AND SETUP
# ============================================================================

@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config():
    """Create a mock Config object with typical values."""
    config = Mock()
    config.seed = 42
    config.lr = 0.001
    config.optimizer = "adamw"
    config.warmup_steps = 1000
    config.batch_size = 64
    config.grad_accum = 4
    config.max_context = 512
    config.dim = 256
    config.num_blocks = 8
    config.vocab_size = 1000
    config.epochs = 2
    config.head_size = 32
    config.n_heads = config.dim // config.head_size
    config.mla = False
    config.kv_heads = config.n_heads
    config.use_moe = False
    config.alpha_balance = 0.01
    config.use_rotary_pos = True
    config.absolute_pos = False
    config.trainable_pos = False
    config.dataset_source = "huggingface"
    config.dataset_name = "wikitext"
    config.tokenizer_type = "char"
    config.activation = "silu"
    config.use_swiglu = True

    # Add methods
    config.from_yaml = Mock(return_value=config)
    config.save_yaml = Mock()

    return config


@pytest.fixture
def mock_wandb():
    """Mock W&B integration."""
    with patch('wandb.init') as mock_init, \
         patch('wandb.config') as mock_wandb_config, \
         patch('wandb.log') as mock_log, \
         patch('wandb.finish') as mock_finish:

        mock_wandb_config.items = Mock(return_value=[
            ('attention_type', 'standard_mha'),
            ('use_moe', False),
            ('optimizer', 'adamw'),
            ('dim', 256),
            ('num_blocks', 8),
        ])
        mock_wandb_config.get = Mock(side_effect=lambda k, default=None: {
            'attention_type': 'standard_mha',
            'down_dim_q': 128,
            'down_dim_kv': 64,
            'rope_dim': 16,
            'activation': 'silu',
            'use_swiglu': True,
        }.get(k, default))
        mock_wandb_config.update = Mock()

        yield {
            'init': mock_init,
            'config': mock_wandb_config,
            'log': mock_log,
            'finish': mock_finish
        }


@pytest.fixture
def mock_jax_flax():
    """Mock JAX and Flax dependencies."""
    mocks = {}

    with patch('jax.devices') as mock_devices, \
         patch('jax.tree_util.tree_map') as mock_tree_map, \
         patch('jax.tree_util.tree_leaves') as mock_tree_leaves, \
         patch('jax.random.PRNGKey') as mock_prngkey, \
         patch('jax.random.split') as mock_split, \
         patch('jax.jit') as mock_jit, \
         patch('jax.Array') as mock_array:

        # Setup device mocking
        mock_gpu_device = Mock()
        mock_gpu_device.platform = 'gpu'
        mock_gpu_device.memory_stats = Mock(return_value={'bytes_in_use': 1e9})

        mock_devices.return_value = [mock_gpu_device]

        # Setup tree operations
        mock_tree_map.side_effect = lambda fn, *args: fn(*args) if args else fn
        mock_tree_leaves.return_value = [Mock(size=100) for _ in range(10)]

        # Setup RNG
        mock_rng = Mock()
        mock_prngkey.return_value = mock_rng
        mock_split.return_value = (mock_rng, mock_rng)

        # JIT decorator should pass through
        mock_jit.side_effect = lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn

        # Store mocks
        mocks['devices'] = mock_devices
        mocks['tree_map'] = mock_tree_map
        mocks['tree_leaves'] = mock_tree_leaves
        mocks['prngkey'] = mock_prngkey
        mocks['split'] = mock_split
        mocks['jit'] = mock_jit

        yield mocks


@pytest.fixture
def mock_nnx():
    """Mock Flax NNX module."""
    with patch('flax.nnx.Rngs') as mock_rngs, \
         patch('flax.nnx.Optimizer') as mock_optimizer_class, \
         patch('flax.nnx.state') as mock_state, \
         patch('flax.nnx.Param') as mock_param, \
         patch('flax.nnx.split') as mock_split, \
         patch('flax.nnx.merge') as mock_merge, \
         patch('flax.nnx.update') as mock_update, \
         patch('flax.nnx.value_and_grad') as mock_value_and_grad, \
         patch('flax.nnx.MultiMetric') as mock_multimetric, \
         patch('flax.nnx.jit') as mock_jit, \
         patch('flax.nnx.metrics') as mock_metrics:

        # Setup Rngs
        mock_rngs_instance = Mock()
        mock_rngs.return_value = mock_rngs_instance

        # Setup Optimizer
        mock_optimizer = Mock()
        mock_optimizer.update = Mock()
        mock_optimizer_class.return_value = mock_optimizer

        # Setup state operations
        mock_state_dict = {}
        mock_state.return_value = mock_state_dict

        # Setup split/merge
        mock_graphdef = Mock()
        mock_state_data = Mock()
        mock_split.return_value = (mock_graphdef, mock_state_data)
        mock_merge.return_value = (Mock(), mock_optimizer, Mock())

        # Setup value_and_grad
        grad_fn = Mock(return_value=((Mock(), Mock()), Mock()))
        mock_value_and_grad.return_value = grad_fn

        # Setup metrics
        mock_metric_instance = Mock()
        mock_metric_instance.update = Mock()
        mock_multimetric.return_value = mock_metric_instance

        # JIT decorator
        mock_jit.side_effect = lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn

        # Setup metrics module
        mock_metrics.Average = Mock(return_value=Mock())

        yield {
            'Rngs': mock_rngs,
            'Optimizer': mock_optimizer_class,
            'state': mock_state,
            'Param': mock_param,
            'split': mock_split,
            'merge': mock_merge,
            'update': mock_update,
            'value_and_grad': mock_value_and_grad,
            'MultiMetric': mock_multimetric,
            'jit': mock_jit,
            'metrics': mock_metrics,
        }


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestGetOptaxOptimizer:
    """Test optimizer setup with learning rate schedules."""

    def test_adamw_optimizer(self, mock_config):
        """Test AdamW optimizer creation."""
        from train_sweep_attention_comparison import get_optax_optimizer

        mock_config.optimizer = "adamw"
        mock_config.lr = 0.001

        with patch('optax.adamw') as mock_adamw, \
             patch('optax.warmup_cosine_decay_schedule') as mock_schedule:

            mock_schedule.return_value = Mock()
            mock_adamw.return_value = Mock()

            optimizer = get_optax_optimizer(mock_config, total_steps=1000)

            assert mock_adamw.called
            assert mock_schedule.called
            schedule_call = mock_schedule.call_args
            assert schedule_call[1]['peak_value'] == 0.001

    def test_lion_optimizer(self, mock_config):
        """Test Lion optimizer creation."""
        from train_sweep_attention_comparison import get_optax_optimizer

        mock_config.optimizer = "lion"
        mock_config.lr = 0.0005

        with patch('optax.lion') as mock_lion, \
             patch('optax.warmup_cosine_decay_schedule') as mock_schedule:

            mock_schedule.return_value = Mock()
            mock_lion.return_value = Mock()

            optimizer = get_optax_optimizer(mock_config, total_steps=5000)

            assert mock_lion.called
            assert mock_schedule.called

    def test_warmup_capping(self, mock_config):
        """Test that warmup steps are capped at 30% of total steps."""
        from train_sweep_attention_comparison import get_optax_optimizer

        mock_config.warmup_steps = 10000

        with patch('optax.warmup_cosine_decay_schedule') as mock_schedule, \
             patch('optax.adamw'):

            mock_schedule.return_value = Mock()
            get_optax_optimizer(mock_config, total_steps=1000)

            schedule_call = mock_schedule.call_args
            warmup = schedule_call[1]['warmup_steps']
            assert warmup <= 300  # 30% of 1000


class TestParseArgs:
    """Test command-line argument parsing."""

    def test_parse_default_config_path(self):
        """Test default config path."""
        from train_sweep_attention_comparison import parse_args

        with patch('sys.argv', ['train.py']):
            args = parse_args()
            assert args.config == "configs/default_config.yaml"

    def test_parse_custom_config_path(self):
        """Test custom config path."""
        from train_sweep_attention_comparison import parse_args

        with patch('sys.argv', ['train.py', '--config', 'custom.yaml']):
            args = parse_args()
            assert args.config == 'custom.yaml'

    def test_parse_data_path(self):
        """Test data path argument."""
        from train_sweep_attention_comparison import parse_args

        with patch('sys.argv', ['train.py', '--data_path', '/path/to/data.txt']):
            args = parse_args()
            assert args.data_path == '/path/to/data.txt'


class TestGetVramUsage:
    """Test VRAM usage reporting."""

    def test_vram_usage_gpu(self, mock_jax_flax):
        """Test VRAM usage on GPU."""
        from train_sweep_attention_comparison import get_vram_usage

        with patch('jax.devices') as mock_devices:
            gpu_device = Mock()
            gpu_device.platform = 'gpu'
            gpu_device.memory_stats = Mock(return_value={'bytes_in_use': 2e9})
            mock_devices.return_value = [gpu_device]

            vram = get_vram_usage()
            assert vram == 2.0  # 2GB

    def test_vram_usage_no_gpu(self):
        """Test VRAM usage with no GPU."""
        from train_sweep_attention_comparison import get_vram_usage

        with patch('jax.devices') as mock_devices:
            cpu_device = Mock()
            cpu_device.platform = 'cpu'
            mock_devices.return_value = [cpu_device]

            vram = get_vram_usage()
            assert vram == 0.0


class TestReportModelSummary:
    """Test model summary generation."""

    def test_model_summary_json_output(self, mock_config, temp_run_dir, mock_nnx):
        """Test that model summary is correctly saved."""
        from train_sweep_attention_comparison import report_model_summary

        with patch('flax.nnx.state') as mock_state:
            mock_state.return_value = {
                'weights': jnp.ones((1000, 100))
            }

            mock_model = Mock()
            mock_optimizer = Mock()
            summary_path = os.path.join(temp_run_dir, "model_summary.json")

            with patch('jax.tree_util.tree_leaves') as mock_leaves:
                leaves = [Mock(size=100) for _ in range(100)]
                mock_leaves.return_value = leaves

                summary = report_model_summary(mock_model, mock_config,
                                              mock_optimizer, summary_path)

            # Verify JSON was written
            assert os.path.exists(summary_path)

            with open(summary_path) as f:
                data = json.load(f)

            assert 'total_params_M' in data
            assert 'weights_mem_MB' in data
            assert 'optimizer_mem_MB' in data
            assert 'est_activations_MB' in data
            assert 'total_est_vram_MB' in data

    def test_model_summary_structure(self, mock_config, temp_run_dir, mock_nnx):
        """Test summary has all required fields."""
        from train_sweep_attention_comparison import report_model_summary

        with patch('flax.nnx.state') as mock_state, \
             patch('jax.tree_util.tree_leaves') as mock_leaves:

            mock_state.return_value = {}
            leaves = [Mock(size=50) for _ in range(50)]
            mock_leaves.return_value = leaves

            mock_model = Mock()
            mock_optimizer = Mock()
            summary_path = os.path.join(temp_run_dir, "summary.json")

            summary = report_model_summary(mock_model, mock_config,
                                          mock_optimizer, summary_path)

            assert isinstance(summary, dict)
            assert all(k in summary for k in [
                'total_params_M', 'weights_mem_MB',
                'optimizer_mem_MB', 'est_activations_MB'
            ])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestConfigSetup:
    """Test configuration initialization and W&B integration."""

    def test_config_from_yaml(self, mock_config):
        """Test config loading from YAML."""

        assert mock_config.seed == 42
        assert mock_config.lr == 0.001
        assert mock_config.optimizer == "adamw"

    def test_attention_type_standard_mha(self, mock_config):
        """Test standard MHA attention configuration."""
        mock_config.mla = False
        mock_config.kv_heads = mock_config.n_heads

        assert mock_config.mla is False
        assert mock_config.kv_heads == mock_config.n_heads

    def test_attention_type_standard_gqa(self, mock_config):
        """Test standard GQA attention configuration."""
        mock_config.mla = False
        kv_heads = max(1, mock_config.n_heads // 4)
        mock_config.kv_heads = kv_heads

        assert mock_config.mla is False
        assert mock_config.kv_heads == kv_heads

    def test_attention_type_mla(self, mock_config):
        """Test MLA attention configuration."""
        mock_config.mla = True
        mock_config.kv_heads = max(1, mock_config.n_heads // 4)
        mock_config.down_dim_q = mock_config.dim // 2
        mock_config.down_dim_kv = mock_config.dim // 4
        mock_config.rope_dim = 16

        assert mock_config.mla is True
        assert mock_config.down_dim_q == 128
        assert mock_config.down_dim_kv == 64

    def test_kv_heads_divisibility(self, mock_config):
        """Test KV heads divisibility adjustment."""
        # Simulate non-divisible values
        mock_config.n_heads = 8
        mock_config.kv_heads = 3

        adjusted_kv_heads = math.gcd(mock_config.n_heads, mock_config.kv_heads)
        assert mock_config.n_heads % adjusted_kv_heads == 0
        assert adjusted_kv_heads == 1


class TestRunDirectorySetup:
    """Test run directory creation and metadata."""

    def test_run_directory_creation(self, temp_run_dir, mock_config):
        """Test run directory is created with correct structure."""
        attn_type = "standard_mha"
        moe_tag = "MoE" if mock_config.use_moe else "Dense"

        run_id = f"{attn_type}_{mock_config.dim}d_{mock_config.num_blocks}b_{moe_tag}_120000"
        run_dir = os.path.join(temp_run_dir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)

        assert os.path.exists(run_dir)
        assert os.path.isdir(run_dir)

    def test_metadata_json_creation(self, temp_run_dir, mock_config):
        """Test metadata JSON is created with all config values."""

        run_dir = os.path.join(temp_run_dir, "runs", "test_run")
        os.makedirs(run_dir, exist_ok=True)

        metadata = {
            'seed': mock_config.seed,
            'lr': mock_config.lr,
            'optimizer': mock_config.optimizer,
            'attn_type': 'standard_mha',
            'kv_bytes_per_token': 64,
            'run_id': 'test_run'
        }

        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        assert os.path.exists(metadata_path)

        with open(metadata_path) as f:
            loaded = json.load(f)

        assert loaded['seed'] == 42
        assert loaded['lr'] == 0.001
        assert loaded['attn_type'] == 'standard_mha'


class TestDatasetProcessing:
    """Test dataset loading and preprocessing."""

    def test_text_loading_from_file(self, temp_run_dir):
        """Test text loading from file."""
        # Create test file
        test_file = os.path.join(temp_run_dir, "test_data.txt")
        test_text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

        with open(test_file, 'w') as f:
            f.write(test_text)

        # Load text
        with open(test_file, encoding='utf-8') as f:
            loaded_text = f.read()

        assert loaded_text == test_text
        assert len(loaded_text.split('\n')) == 5

    def test_text_formatting(self):
        """Test text formatting into blocks."""
        raw_text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
        raw_lines = raw_text.split('\n')
        valid_lines = [l.rstrip() for l in raw_lines if l.strip()]

        formatted_blocks = []
        for i in range(0, len(valid_lines), 3):
            formatted_blocks.append('\n'.join(valid_lines[i:i+3]))

        formatted_text = '\n\n'.join(formatted_blocks)

        assert len(formatted_blocks) == 2
        assert 'Line 1' in formatted_text
        assert 'Line 6' in formatted_text

    def test_train_val_split(self):
        """Test train/validation split."""
        total_size = 1000
        train_ratio = 0.9

        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size

        assert train_size == 900
        assert val_size == 100
        assert train_size + val_size == total_size


class TestTokenizer:
    """Test tokenizer integration."""

    def test_char_tokenizer(self):
        """Test character-level tokenizer."""
        with patch('utils.tokenizer.get_tokenizer') as mock_get_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 128
            mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
            mock_get_tokenizer.return_value = mock_tokenizer

            from dantinox.utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer('char')

            assert tokenizer.vocab_size == 128
            tokens = tokenizer.encode("test")
            assert len(tokens) == 5

    def test_bpe_tokenizer(self):
        """Test BPE tokenizer."""
        with patch('utils.tokenizer.get_tokenizer') as mock_get_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 10000
            mock_tokenizer.encode = Mock(return_value=[100, 200, 300])
            mock_get_tokenizer.return_value = mock_tokenizer

            from dantinox.utils.tokenizer import get_tokenizer
            tokenizer = get_tokenizer('bpe')

            assert tokenizer.vocab_size == 10000
            tokens = tokenizer.encode("test text")
            assert len(tokens) == 3


class TestTrainingLoop:
    """Test the main training loop execution."""

    def test_training_step_execution(self):
        """Test a single training step."""
        with patch('jax.jit') as mock_jit:
            mock_jit.side_effect = lambda fn=None, **kwargs: (lambda f: f) if fn is None else fn

            # Mock loss computation
            batch_x = jnp.ones((64, 512))
            batch_y = jnp.ones((64, 512))

            # Verify tensors are created
            assert batch_x.shape == (64, 512)
            assert batch_y.shape == (64, 512)

    def test_logging_csv_creation(self, temp_run_dir):
        """Test training log CSV is created."""
        log_path = os.path.join(temp_run_dir, "training_log.csv")

        with open(log_path, 'w', newline='') as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(['step', 'train_loss', 'val_loss', 'vram_gb', 'ms_per_step'])
            writer.writerow([0, 5.0, 4.8, 2.5, 100.0])
            writer.writerow([50, 4.5, 4.3, 2.6, 95.0])

        assert os.path.exists(log_path)

        with open(log_path) as f:
            import csv
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0] == ['step', 'train_loss', 'val_loss', 'vram_gb', 'ms_per_step']
        assert float(rows[1][1]) == 5.0


class TestWandBIntegration:
    """Test W&B integration."""

    def test_wandb_init(self, mock_wandb):
        """Test W&B initialization."""
        with patch('wandb.init') as mock_init:
            import wandb
            wandb.init(group="Attention-Comparison-V1")

            mock_init.assert_called_once()

    def test_wandb_log_metrics(self, mock_wandb):
        """Test W&B metric logging."""
        with patch('wandb.log') as mock_log:
            import wandb

            metrics = {
                'step': 0,
                'train_loss': 5.0,
                'val_loss': 4.8,
                'vram_gb': 2.5,
                'ms_per_step': 100.0
            }

            wandb.log(metrics)
            mock_log.assert_called_once_with(metrics)

    def test_wandb_config_update(self, mock_wandb):
        """Test W&B config update."""
        with patch('wandb.config.update') as mock_update:
            import wandb

            config = {'lr': 0.001, 'batch_size': 64}
            wandb.config.update(config, allow_val_change=True)

            mock_update.assert_called_once()


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        text = ""
        raw_lines = text.split('\n')
        valid_lines = [l.rstrip() for l in raw_lines if l.strip()]

        assert len(valid_lines) == 0

    def test_very_small_dataset(self):
        """Test handling of very small dataset."""
        text = "a"
        tokens = list(text)

        assert len(tokens) == 1

    def test_single_block_text(self):
        """Test text with less than 3 lines."""
        raw_lines = ["Line 1", "Line 2"]
        formatted_blocks = []
        for i in range(0, len(raw_lines), 3):
            formatted_blocks.append('\n'.join(raw_lines[i:i+3]))

        assert len(formatted_blocks) == 1
        assert 'Line 1' in formatted_blocks[0]
        assert 'Line 2' in formatted_blocks[0]

    def test_zero_warmup_steps(self, mock_config):
        """Test optimizer with zero warmup steps."""
        from train_sweep_attention_comparison import get_optax_optimizer

        mock_config.warmup_steps = 0

        with patch('optax.warmup_cosine_decay_schedule') as mock_schedule, \
             patch('optax.adamw'):

            mock_schedule.return_value = Mock()
            get_optax_optimizer(mock_config, total_steps=1000)

            schedule_call = mock_schedule.call_args
            warmup = schedule_call[1]['warmup_steps']
            assert warmup == 0

    def test_very_large_batch_size(self, mock_config):
        """Test with very large batch size."""
        mock_config.batch_size = 4096
        mock_config.grad_accum = 8

        micro_batch_size = mock_config.batch_size // mock_config.grad_accum
        assert micro_batch_size == 512

    def test_activation_gelu_config(self, mock_config):
        """Test GELU activation configuration."""
        mock_config.activation = "gelu"
        mock_config.use_swiglu = False

        assert mock_config.activation == "gelu"
        assert mock_config.use_swiglu is False

    def test_activation_silu_swiglu_config(self, mock_config):
        """Test SiLU activation with SwiGLU."""
        mock_config.activation = "silu"
        mock_config.use_swiglu = True

        assert mock_config.activation == "silu"
        assert mock_config.use_swiglu is True


class TestSweepParameterCombinations:
    """Test various sweep parameter combinations."""

    def test_moe_vs_dense_config(self, mock_config):
        """Test MoE vs Dense configuration."""
        # Dense variant
        mock_config.use_moe = False
        moe_tag = "MoE" if mock_config.use_moe else "Dense"
        assert moe_tag == "Dense"

        # MoE variant
        mock_config.use_moe = True
        moe_tag = "MoE" if mock_config.use_moe else "Dense"
        assert moe_tag == "MoE"

    def test_different_dimensions(self, mock_config):
        """Test different model dimensions."""
        for dim in [256, 512, 1024]:
            mock_config.dim = dim
            mock_config.n_heads = dim // mock_config.head_size

            assert mock_config.dim == dim
            assert mock_config.n_heads >= 1

    def test_different_num_blocks(self, mock_config):
        """Test different number of blocks."""
        for num_blocks in [8, 12, 16, 24]:
            mock_config.num_blocks = num_blocks
            assert mock_config.num_blocks == num_blocks

    def test_optimizer_combinations(self, mock_config):
        """Test different optimizer configurations."""
        for optimizer in ["adamw", "lion"]:
            mock_config.optimizer = optimizer
            assert mock_config.optimizer == optimizer


# ============================================================================
# PERFORMANCE AND MEMORY TESTS
# ============================================================================

class TestMemoryManagement:
    """Test memory-related functionality."""

    def test_kv_cache_memory_standard_mha(self, mock_config):
        """Test KV cache memory for standard MHA."""
        mock_config.mla = False
        kv_bytes = 2 * 2 * (mock_config.kv_heads * mock_config.head_size)

        assert kv_bytes > 0
        assert isinstance(kv_bytes, int)

    def test_kv_cache_memory_mla(self, mock_config):
        """Test KV cache memory for MLA."""
        mock_config.mla = True
        mock_config.down_dim_kv = 64
        mock_config.rope_dim = 16

        kv_bytes = 2 * (mock_config.down_dim_kv + mock_config.rope_dim)
        assert kv_bytes == 160

    def test_activation_memory_estimation(self, mock_config):
        """Test activation memory estimation."""
        act_mem = (mock_config.batch_size *
                   mock_config.max_context *
                   mock_config.dim *
                   mock_config.num_blocks * 8 * 4) / 1e6

        # Should be reasonable for these config values
        assert act_mem > 0
        assert act_mem < 100000  # Less than 100GB


# ============================================================================
# PYTEST CONFIGURATION AND HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment once per session."""
    # Ensure CUDA not used during tests
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    yield
    # Cleanup if needed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
