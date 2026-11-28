(function () {
  function initBlogHome() {
    var root = document.getElementById('blog-home');
    if (!root) return;

    var cards = Array.prototype.slice.call(root.querySelectorAll('.bh-card'));
    var monthGroups = Array.prototype.slice.call(root.querySelectorAll('[data-month-group]'));
    var navButtons = Array.prototype.slice.call(root.querySelectorAll('[data-filter-cat]'));
    var topicChips = Array.prototype.slice.call(root.querySelectorAll('[data-cat-jump]'));
    var viewButtons = Array.prototype.slice.call(root.querySelectorAll('[data-view]'));
    var timelineItems = Array.prototype.slice.call(root.querySelectorAll('[data-timeline-item]'));
    var timelineYears = Array.prototype.slice.call(root.querySelectorAll('[data-timeline-year]'));
    var timelineLinks = Array.prototype.slice.call(root.querySelectorAll('[data-month-target]'));

    var searchInput = document.getElementById('bh-filter-input');
    var clearBtn = document.getElementById('bh-filter-clear');
    var pinnedSection = document.getElementById('bh-pinned-section');
    var activeLabel = document.getElementById('bh-active-label');
    var activeCount = document.getElementById('bh-active-count');
    var resetBtn = document.getElementById('bh-reset-btn');
    var emptyResetBtn = document.getElementById('bh-empty-reset');
    var emptyState = document.getElementById('bh-empty');

    var labelMap = {
      all: '全部文章',
      'tech-notes': '技术解析',
      papers: '全部论文精读',
    };

    navButtons.forEach(function (btn) {
      var slug = btn.getAttribute('data-filter-cat');
      if (!slug || labelMap[slug]) return;
      var labelNode = btn.querySelector('.bh-nav-label');
      if (labelNode) {
        labelMap[slug] = labelNode.textContent.trim();
      }
    });

    var state = {
      cat: 'all',
      query: '',
      view: 'card',
    };

    try {
      var savedView = localStorage.getItem('blog-home-view');
      if (savedView === 'compact' || savedView === 'card') {
        state.view = savedView;
      }
    } catch (_) {}

    function parseHashCat() {
      var hash = window.location.hash || '';
      var match = hash.match(/cat=([^&]+)/);
      if (!match) return null;
      var decoded = decodeURIComponent(match[1]);
      return labelMap[decoded] ? decoded : null;
    }

    function updateHash(cat) {
      if (!window.history || !window.history.replaceState) return;
      var base = window.location.pathname + window.location.search;
      if (cat === 'all') {
        if (window.location.hash && window.location.hash.indexOf('cat=') !== -1) {
          window.history.replaceState(null, '', base);
        }
      } else {
        window.history.replaceState(null, '', base + '#cat=' + encodeURIComponent(cat));
      }
    }

    function setView(view) {
      state.view = view === 'compact' ? 'compact' : 'card';
      root.classList.toggle('is-compact', state.view === 'compact');
      viewButtons.forEach(function (btn) {
        btn.classList.toggle('is-active', btn.getAttribute('data-view') === state.view);
      });
      try {
        localStorage.setItem('blog-home-view', state.view);
      } catch (_) {}
    }

    function matchesCard(card) {
      var kind = card.getAttribute('data-kind');
      var cat = card.getAttribute('data-cat');

      if (state.cat === 'papers') {
        if (kind !== 'paper') return false;
      } else if (state.cat !== 'all') {
        if (cat !== state.cat) return false;
      }

      if (state.query) {
        var haystack = card.getAttribute('data-search') || '';
        var tokens = state.query.split(/\s+/).filter(Boolean);
        for (var i = 0; i < tokens.length; i++) {
          if (haystack.indexOf(tokens[i]) === -1) return false;
        }
      }
      return true;
    }

    function applyFilter(options) {
      var syncHash = !options || options.syncHash !== false;
      var visibleTotal = 0;
      var monthCounts = {};

      cards.forEach(function (card) {
        var visible = matchesCard(card);
        card.hidden = !visible;
        if (visible) {
          visibleTotal += 1;
          var mKey = card.getAttribute('data-month');
          monthCounts[mKey] = (monthCounts[mKey] || 0) + 1;
        }
      });

      var visibleYears = {};
      monthGroups.forEach(function (group) {
        var mKey = group.getAttribute('data-month-group');
        var count = monthCounts[mKey] || 0;
        group.hidden = count === 0;
        var countEl = group.querySelector('[data-group-count]');
        if (countEl) countEl.textContent = count + ' 篇';
        if (count > 0) {
          visibleYears[group.getAttribute('data-year')] = true;
        }
      });

      timelineItems.forEach(function (li) {
        var mKey = li.getAttribute('data-timeline-item');
        var count = monthCounts[mKey] || 0;
        li.hidden = count === 0;
        var countEl = li.querySelector('[data-month-count]');
        if (countEl) countEl.textContent = String(count);
      });

      timelineYears.forEach(function (yearEl) {
        var y = yearEl.getAttribute('data-timeline-year');
        yearEl.hidden = !visibleYears[y];
      });

      if (pinnedSection) {
        var showPinned = !state.query && (state.cat === 'all' || state.cat === 'tech-notes');
        pinnedSection.hidden = !showPinned;
      }

      navButtons.forEach(function (btn) {
        var isActive = btn.getAttribute('data-filter-cat') === state.cat;
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });

      var isFiltered = state.cat !== 'all' || Boolean(state.query);
      if (activeLabel) {
        activeLabel.textContent = state.query
          ? (labelMap[state.cat] || '筛选结果') + ' · 搜索 “' + state.query + '”'
          : (labelMap[state.cat] || '全部文章');
      }
      if (activeCount) {
        activeCount.textContent = visibleTotal + ' 篇';
      }
      if (resetBtn) {
        resetBtn.hidden = !isFiltered;
      }
      if (clearBtn) {
        clearBtn.hidden = !state.query;
      }
      if (emptyState) {
        emptyState.hidden = visibleTotal > 0;
      }

      if (syncHash) {
        updateHash(state.cat);
      }

      updateActiveTimelineFromScroll();
    }

    function setCategory(slug, options) {
      if (!labelMap[slug]) slug = 'all';
      state.cat = slug;
      applyFilter(options);
    }

    function resetAll() {
      state.cat = 'all';
      state.query = '';
      if (searchInput) searchInput.value = '';
      applyFilter();
    }

    // Bind category buttons (desktop sidebar + mobile chips)
    navButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var slug = btn.getAttribute('data-filter-cat');
        setCategory(slug);
        if (window.scrollY > root.offsetTop + 180) {
          root.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });

    // Clickable topic chips inside post cards
    topicChips.forEach(function (chip) {
      chip.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var slug = chip.getAttribute('data-cat-jump');
        setCategory(slug);
        root.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });

    // View toggle
    viewButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        setView(btn.getAttribute('data-view'));
      });
    });

    // Filter search input
    if (searchInput) {
      searchInput.addEventListener('input', function () {
        state.query = searchInput.value.trim().toLowerCase();
        applyFilter({ syncHash: false });
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        state.query = '';
        if (searchInput) {
          searchInput.value = '';
          searchInput.focus();
        }
        applyFilter({ syncHash: false });
      });
    }

    if (resetBtn) resetBtn.addEventListener('click', resetAll);
    if (emptyResetBtn) emptyResetBtn.addEventListener('click', resetAll);

    // Keyboard shortcut "/" to focus quick filter
    document.addEventListener('keydown', function (e) {
      if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
      var tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || (e.target && e.target.isContentEditable)) return;
      if (searchInput) {
        e.preventDefault();
        searchInput.focus();
      }
    });

    var topbar = root.querySelector('.bh-topbar');
    var navbar = document.getElementById('navbar');

    function syncStickyOffsets() {
      var collapsedNavHeight =
        navbar && navbar.classList.contains('top-nav-collapse')
          ? Math.round(navbar.getBoundingClientRect().height)
          : 50;
      var topbarTop = Math.max(52, Math.min(collapsedNavHeight + 8, 66));
      root.style.setProperty('--bh-topbar-top', topbarTop + 'px');
      if (topbar) {
        var topbarHeight = Math.round(topbar.getBoundingClientRect().height);
        root.style.setProperty('--bh-sticky-offset', topbarTop + topbarHeight + 14 + 'px');
      }
    }

    function updateTopbarStuck() {
      if (!topbar) return;
      var topPx = parseFloat(getComputedStyle(topbar).top) || 58;
      var isStuck = topbar.getBoundingClientRect().top <= topPx + 1;
      topbar.classList.toggle('is-stuck', isStuck);
    }

    // Timeline smooth scroll + ScrollSpy
    function setCurrentMonth(mKey) {
      timelineLinks.forEach(function (link) {
        link.classList.toggle('is-current', link.getAttribute('data-month-target') === mKey);
      });
    }

    function updateActiveTimelineFromScroll() {
      var threshold = topbar ? topbar.getBoundingClientRect().bottom + 36 : 160;
      var activeKey = null;
      for (var i = 0; i < monthGroups.length; i++) {
        var group = monthGroups[i];
        if (group.hidden) continue;
        var rect = group.getBoundingClientRect();
        if (activeKey === null) {
          activeKey = group.getAttribute('data-month-group');
        }
        if (rect.top <= threshold) {
          activeKey = group.getAttribute('data-month-group');
        } else {
          break;
        }
      }
      if (activeKey) setCurrentMonth(activeKey);
    }

    timelineLinks.forEach(function (link) {
      link.addEventListener('click', function (e) {
        var mKey = link.getAttribute('data-month-target');
        var target = document.getElementById('month-' + mKey);
        if (!target || target.hidden) return;
        e.preventDefault();
        setCurrentMonth(mKey);
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });

    var ticking = false;
    window.addEventListener(
      'scroll',
      function () {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(function () {
          updateTopbarStuck();
          updateActiveTimelineFromScroll();
          ticking = false;
        });
      },
      { passive: true }
    );

    window.addEventListener('resize', function () {
      syncStickyOffsets();
      updateTopbarStuck();
    });

    window.addEventListener('hashchange', function () {
      var hashCat = parseHashCat();
      if (hashCat && hashCat !== state.cat) {
        setCategory(hashCat, { syncHash: false });
      } else if (!window.location.hash && state.cat !== 'all') {
        setCategory('all', { syncHash: false });
      }
    });

    // Initialize
    syncStickyOffsets();
    setView(state.view);
    var initialCat = parseHashCat();
    if (initialCat) {
      state.cat = initialCat;
    }
    applyFilter({ syncHash: false });
    updateTopbarStuck();

    // When returning from a post breadcrumb (/#cat=...), glide past the 100vh hero to the workspace.
    if (initialCat) {
      window.requestAnimationFrame(function () {
        syncStickyOffsets();
        root.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBlogHome);
  } else {
    initBlogHome();
  }
})();
