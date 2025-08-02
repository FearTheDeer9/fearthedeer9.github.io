document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit to ensure all scripts are loaded
    setTimeout(() => {
        const filterButtons = document.querySelectorAll('.nav-btn');
        const posts = document.querySelectorAll('article.post');
    
        // Show a message when no posts match the filter
        function showNoPostsMessage(show = true) {
            let messageDiv = document.querySelector('.no-posts-message');
            if (show) {
                if (!messageDiv) {
                    messageDiv = document.createElement('div');
                    messageDiv.className = 'no-posts-message';
                    messageDiv.innerHTML = '<p style="text-align: center; color: #666; margin: 2rem 0;">No posts found for this category. Posts may be on other pages.</p>';
                    document.querySelector('.posts-container').appendChild(messageDiv);
                }
                messageDiv.style.display = 'block';
            } else if (messageDiv) {
                messageDiv.style.display = 'none';
            }
        }
        
        filterButtons.forEach(button => {
            button.addEventListener('click', () => {
                const filter = button.dataset.filter;
                const heroSection = document.querySelector('.hero-section');
                const featuredSection = document.querySelector('.featured-section');
                const postsTitle = document.getElementById('posts-title');
                const searchInput = document.getElementById('search-input');
                
                // Clear search when filter is clicked
                if (searchInput) {
                    searchInput.value = '';
                }
                
                // Update sections visibility based on filter
                if (filter === 'all') {
                    // Show everything for "All Posts"
                    if (heroSection) heroSection.style.display = '';
                    if (featuredSection) featuredSection.style.display = '';
                    if (postsTitle) postsTitle.textContent = 'All Posts';
                } else {
                    // Hide hero and featured for filtered views
                    if (heroSection) heroSection.style.display = 'none';
                    if (featuredSection) featuredSection.style.display = 'none';
                    
                    // Update title based on filter
                    if (postsTitle) {
                        if (filter === 'paper-summary') {
                            postsTitle.textContent = 'Paper Summaries';
                        } else if (filter === 'short-form') {
                            postsTitle.textContent = 'Short-form Posts';
                        }
                    }
                }
                
                let visibleCount = 0;
                
                posts.forEach(post => {
                    const categories = post.dataset.categories?.split(' ').filter(cat => cat.trim()) || [];
                    const shouldShow = filter === 'all' || categories.includes(filter);
                    
                    if (shouldShow) {
                        post.style.display = '';  // Reset to default (grid item)
                        post.classList.remove('hidden');
                        visibleCount++;
                    } else {
                        post.style.display = 'none';
                        post.classList.add('hidden');
                    }
                });
                
                // Show message if no posts are visible and it's not "all"
                showNoPostsMessage(visibleCount === 0 && filter !== 'all');
                
                // Update active button state
                filterButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Smooth scroll to posts section when filtering
                if (filter !== 'all') {
                    document.querySelector('.posts-container').scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'start' 
                    });
                }
            });
        });
    }, 100); // Small delay to ensure DOM is ready
});