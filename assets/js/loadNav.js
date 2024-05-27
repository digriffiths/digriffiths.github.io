window.onload = function() {
    const navElement = document.getElementById('nav');
    // Adjust the path as necessary, e.g., './nav.html' or '/repository/nav.html'
    fetch('./nav.html')
        .then(response => response.text())
        .then(html => {
            navElement.innerHTML = html;
            const navLinks = document.querySelectorAll('#nav .links li');
            const currentPath = window.location.pathname.split('/').pop();

            navLinks.forEach(link => {
                if (link.querySelector('a').getAttribute('href').includes(currentPath)) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            });
        })
        .catch(error => console.error('Error loading the navigation bar:', error));
};