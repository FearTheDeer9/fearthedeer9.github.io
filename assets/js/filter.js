document.addEventListener('DOMContentLoaded', () => {
    console.log('Filter script loaded!');
    
    const filterButtons = document.querySelectorAll('.nav-btn');
    const posts = document.querySelectorAll('article.post');  // Make sure we're selecting articles with class 'post'
    
    console.log('Found buttons:', filterButtons.length);
    console.log('Found posts:', posts.length);
    
    filterButtons.forEach(button => {
      button.addEventListener('click', () => {
        const filter = button.dataset.filter;
        console.log('Filter clicked:', filter);
        
        // Log each post and its categories for debugging
        posts.forEach(post => {
          const categories = post.dataset.categories?.split(' ') || [];
          console.log('Post:', post);
          console.log('Post categories:', categories);
          console.log('Should show:', filter === 'all' || categories.includes(filter));
          
          // Force style changes directly for debugging
          if (filter === 'all' || categories.includes(filter)) {
            post.style.display = 'block';
            post.classList.remove('hidden');
          } else {
            post.style.display = 'none';
            post.classList.add('hidden');
          }
        });
        
        // Update active button state
        filterButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
      });
    });
  });