/* Apply the saved theme before CSS paints. Classic script on purpose: modules are deferred. */
(function () {
  try {
    var choice = localStorage.getItem('ragmate_theme') || 'system';
    var systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    var dark = choice === 'dark' || (choice !== 'light' && systemDark);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    document.documentElement.dataset.themeChoice = choice;
  } catch (err) {
    document.documentElement.dataset.theme = 'light';
    document.documentElement.dataset.themeChoice = 'light';
  }
})();
