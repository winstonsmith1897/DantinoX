window.MathJax = {
  tex: {
    inlineMath:  [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

/*
 * Re-typeset on every navigation event (instant navigation, tab reveals,
 * details expand). Without this, LaTeX inside tabs/admonitions/details
 * stays as raw text after the first page load.
 */
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.typeset();
});
