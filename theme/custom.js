// Custom Theme JavaScript

// Scroll to Top Button
document.addEventListener('DOMContentLoaded', function() {
  var scrollBtn = document.createElement('button');
  scrollBtn.className = 'scroll-to-top';
  scrollBtn.innerHTML = '&#9650;';
  scrollBtn.title = 'Scroll to top';
  scrollBtn.setAttribute('aria-label', 'Scroll to top');
  document.body.appendChild(scrollBtn);

  window.addEventListener('scroll', function() {
    if (window.scrollY > 300) {
      scrollBtn.classList.add('visible');
    } else {
      scrollBtn.classList.remove('visible');
    }
  });

  scrollBtn.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});
