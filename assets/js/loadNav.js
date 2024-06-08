window.onload = function() {
    const navElement = document.getElementById('nav');
    fetch('./nav.html')
        .then(response => response.text())
        .then(html => {
            navElement.innerHTML = html;
            const navLinks = document.querySelectorAll('#nav ul li a'); // Corrected selector
            const currentPath = window.location.pathname.split('/').pop();

            navLinks.forEach(link => {
                if (link.getAttribute('href').includes(currentPath)) {
                    link.parentElement.classList.add('active'); // Add class to li, not a
                } else {
                    link.parentElement.classList.remove('active');
                }
            });
        })
        .catch(error => console.error('Error loading the navigation bar:', error));
};