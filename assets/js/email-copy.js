document.addEventListener('DOMContentLoaded', () => {
    const emailButton = document.querySelector('.email-copy');
    
    if (emailButton) {
        emailButton.addEventListener('click', async () => {
            const email = emailButton.dataset.email;
            const emailText = emailButton.querySelector('.email-text');
            const originalText = emailText.textContent;
            
            try {
                // Try modern clipboard API first
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    await navigator.clipboard.writeText(email);
                } else {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = email;
                    textArea.style.position = 'fixed';
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                }
                
                // Visual feedback
                emailText.textContent = 'Copied!';
                emailButton.classList.add('copied');
                
                // Reset after 2 seconds
                setTimeout(() => {
                    emailText.textContent = originalText;
                    emailButton.classList.remove('copied');
                }, 2000);
                
            } catch (err) {
                // If all else fails, show the email
                emailText.textContent = email;
                setTimeout(() => {
                    emailText.textContent = originalText;
                }, 3000);
            }
        });
    }
});