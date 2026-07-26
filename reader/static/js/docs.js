(() => {
  const links = [...document.querySelectorAll('.docs-toc a')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);

  links.forEach(link => {
    link.addEventListener('click', event => {
      const target = document.querySelector(link.getAttribute('href'));
      if (!target) return;
      event.preventDefault();
      const header = document.querySelector('.topnav');
      const offset = (header?.getBoundingClientRect().height || 0) + 18;
      const top = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
      history.replaceState(null, '', link.getAttribute('href'));
    });
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach(link => link.classList.toggle(
        'active', link.getAttribute('href') === `#${visible.target.id}`
      ));
    }, { rootMargin: '-18% 0px -68% 0px', threshold: [0, 0.1, 0.5] });
    sections.forEach(section => observer.observe(section));
  }

  document.querySelectorAll('.copy-code').forEach(button => {
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy || '');
        const original = button.textContent;
        button.textContent = 'Másolva';
        button.classList.add('copied');
        window.setTimeout(() => {
          button.textContent = original;
          button.classList.remove('copied');
        }, 1600);
      } catch (_) {
        button.textContent = 'Ctrl+C';
      }
    });
  });
})();
