// Dark mode toggle functionality
(function() {
  // Check for saved theme preference or default to 'light'
  const currentTheme = localStorage.getItem('theme') || 'light';
  
  // Apply theme on page load
  if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
  
  // Create and add toggle button
  function createToggleButton() {
    const toggleButton = document.createElement('button');
    toggleButton.className = 'theme-toggle';
    toggleButton.setAttribute('aria-label', 'Toggle dark mode');
    toggleButton.innerHTML = `
      <span class="sun-icon">☀️</span>
      <span class="moon-icon">🌙</span>
    `;
    
    // Add click event listener
    toggleButton.addEventListener('click', function() {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      // Apply new theme
      if (newTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      
      // Save theme preference
      localStorage.setItem('theme', newTheme);
    });
    
    // Add to page
    document.body.appendChild(toggleButton);
  }
  
  // Initialize toggle button when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createToggleButton);
  } else {
    createToggleButton();
  }
})();