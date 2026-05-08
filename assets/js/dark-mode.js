(function() {
  var currentTheme = localStorage.getItem('theme') || 'light';
  if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  function addToggle() {
    var nav = document.querySelector('.greedy-nav');
    if (!nav) return;

    var btn = document.createElement('button');
    btn.className = 'theme-toggle-nav';
    btn.setAttribute('aria-label', 'Toggle dark mode');
    btn.textContent = currentTheme === 'dark' ? '☀️' : '🌙';

    btn.addEventListener('click', function() {
      var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
        btn.textContent = '🌙';
      } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        btn.textContent = '☀️';
      }
    });

    nav.appendChild(btn);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addToggle);
  } else {
    addToggle();
  }
})();
