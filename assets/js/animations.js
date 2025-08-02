document.addEventListener('DOMContentLoaded', () => {
  // Back to top button
  const backToTopButton = document.createElement('div');
  backToTopButton.className = 'back-to-top';
  backToTopButton.setAttribute('aria-label', 'Back to top');
  document.body.appendChild(backToTopButton);
  
  // Show/hide back to top button
  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
      backToTopButton.classList.add('visible');
    } else {
      backToTopButton.classList.remove('visible');
    }
  });
  
  // Scroll to top functionality
  backToTopButton.addEventListener('click', () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
  
  // Scroll reveal for elements
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const scrollObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        scrollObserver.unobserve(entry.target);
      }
    });
  }, observerOptions);
  
  // Add scroll reveal to featured posts and section titles
  document.querySelectorAll('.featured-post, .section-title').forEach(el => {
    el.classList.add('scroll-reveal');
    scrollObserver.observe(el);
  });
  
  // Smooth anchor scrolling
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
  
  // Add loading states to links (excluding mailto and external links)
  document.querySelectorAll('a:not([href^="#"]):not([href^="mailto:"]):not([target="_blank"])').forEach(link => {
    link.addEventListener('click', function(e) {
      if (!e.ctrlKey && !e.metaKey && !this.target && this.href.startsWith(window.location.origin)) {
        const pageTransition = document.createElement('div');
        pageTransition.className = 'page-transition';
        document.body.appendChild(pageTransition);
        
        setTimeout(() => {
          pageTransition.classList.add('active');
        }, 10);
      }
    });
  });
  
  // Parallax effect for hero background
  const heroBackground = document.querySelector('.hero-background');
  if (heroBackground) {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      const parallax = scrolled * 0.5;
      heroBackground.style.transform = `translateY(${parallax}px)`;
    });
  }
  
  // Add hover effect to cards
  const posts = document.querySelectorAll('.post, .featured-post');
  posts.forEach(post => {
    post.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-4px)';
    });
    
    post.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
    });
  });
});