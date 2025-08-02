document.addEventListener('DOMContentLoaded', () => {
  // Search functionality
  const searchInput = document.getElementById('search-input');
  const posts = document.querySelectorAll('article.post');
  const postsContainer = document.querySelector('.posts-container');
  
  // Sticky navbar effect
  const navbar = document.querySelector('.navbar');
  let lastScroll = 0;
  
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
    
    lastScroll = currentScroll;
  });
  
  // Mobile menu toggle
  const mobileToggle = document.querySelector('.mobile-menu-toggle');
  const navFilters = document.querySelector('.nav-filters');
  
  mobileToggle?.addEventListener('click', () => {
    navFilters.classList.toggle('active');
  });
  
  // Search functionality
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const searchTerm = e.target.value.toLowerCase().trim();
      const heroSection = document.querySelector('.hero-section');
      const featuredSection = document.querySelector('.featured-section');
      const postsTitle = document.getElementById('posts-title');
      const filterButtons = document.querySelectorAll('.nav-btn');
      let hasResults = false;
      
      // Update visibility and title based on search
      if (searchTerm === '') {
        // No search - restore default view
        if (heroSection) heroSection.style.display = '';
        if (featuredSection) featuredSection.style.display = '';
        if (postsTitle) postsTitle.textContent = 'All Posts';
        
        // Reset filter to "All Posts"
        filterButtons.forEach(btn => {
          if (btn.dataset.filter === 'all') {
            btn.classList.add('active');
          } else {
            btn.classList.remove('active');
          }
        });
      } else {
        // Active search - hide hero/featured and update title
        if (heroSection) heroSection.style.display = 'none';
        if (featuredSection) featuredSection.style.display = 'none';
        if (postsTitle) postsTitle.textContent = `Search Results for "${searchTerm}"`;
        
        // Remove active state from all filter buttons during search
        filterButtons.forEach(btn => btn.classList.remove('active'));
        
        // Smooth scroll to posts when starting search
        if (searchTerm.length === 1) { // Only on first character
          document.querySelector('.posts-container')?.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
          });
        }
      }
      
      posts.forEach(post => {
        const title = post.querySelector('.post-title')?.textContent.toLowerCase() || '';
        const excerpt = post.querySelector('.post-excerpt')?.textContent.toLowerCase() || '';
        const tags = post.dataset.categories?.toLowerCase() || '';
        
        if (searchTerm === '' || 
            title.includes(searchTerm) || 
            excerpt.includes(searchTerm) || 
            tags.includes(searchTerm)) {
          post.style.display = '';  // Reset to default (grid item)
          post.classList.remove('hidden');
          hasResults = true;
          
          // Add highlight effect for search matches
          if (searchTerm !== '') {
            post.classList.add('search-match');
          } else {
            post.classList.remove('search-match');
          }
        } else {
          post.style.display = 'none';
          post.classList.add('hidden');
        }
      });
      
      // Show no results message
      let noResultsMsg = document.querySelector('.no-results');
      if (!hasResults && searchTerm !== '') {
        if (!noResultsMsg) {
          noResultsMsg = document.createElement('div');
          noResultsMsg.className = 'no-results';
          noResultsMsg.textContent = `No posts found for "${searchTerm}"`;
          postsContainer.appendChild(noResultsMsg);
        } else {
          noResultsMsg.textContent = `No posts found for "${searchTerm}"`;
          noResultsMsg.style.display = 'block';
        }
      } else if (noResultsMsg) {
        noResultsMsg.style.display = 'none';
      }
    });
    
    // Clear search on Escape key
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        searchInput.value = '';
        searchInput.dispatchEvent(new Event('input'));
      }
    });
  }
  
  // Keyboard shortcut for search (Cmd/Ctrl + K)
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      searchInput?.focus();
    }
  });
});