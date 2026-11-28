(function () {
  function initBlogArchive() {
    var root = document.getElementById('blog-archive');
    if (!root) return;

    var tbody = document.getElementById('ba-table-body');
    var rows = Array.prototype.slice.call(tbody.querySelectorAll('.ba-row'));
    var yearTabs = Array.prototype.slice.call(root.querySelectorAll('[data-hm-tab]'));
    var yearPanels = Array.prototype.slice.call(root.querySelectorAll('[data-hm-year]'));
    var hmCells = Array.prototype.slice.call(root.querySelectorAll('.ba-hm-cell[data-hm-date]'));
    var catBars = Array.prototype.slice.call(root.querySelectorAll('[data-bar-cat]'));
    var monthBars = Array.prototype.slice.call(root.querySelectorAll('[data-bar-month]'));
    var kindTabs = Array.prototype.slice.call(root.querySelectorAll('[data-kind-tab]'));
    var orgPills = Array.prototype.slice.call(root.querySelectorAll('[data-org-filter]'));
    var rowCatChips = Array.prototype.slice.call(root.querySelectorAll('[data-row-cat]'));

    var searchInput = document.getElementById('ba-search-input');
    var searchClear = document.getElementById('ba-search-clear');
    var catSelect = document.getElementById('ba-cat-select');
    var sortSelect = document.getElementById('ba-sort-select');
    var resultCount = document.getElementById('ba-result-count');
    var filterTags = document.getElementById('ba-filter-tags');
    var resetBtn = document.getElementById('ba-reset-btn');
    var emptyResetBtn = document.getElementById('ba-empty-reset');
    var emptyState = document.getElementById('ba-empty');
    var tableSection = document.getElementById('ba-table-section');
    var tooltip = document.getElementById('ba-tooltip');

    var state = {
      kind: 'paper', // 'paper' | 'note' | 'all'
      cat: 'all',
      org: '',
      date: '',
      month: '',
      query: '',
      sort: 'date-desc',
    };

    // Heatmap year tabs
    function activateYear(year) {
      yearTabs.forEach(function (tab) {
        var active = tab.getAttribute('data-hm-tab') === String(year);
        tab.classList.toggle('is-active', active);
        tab.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      yearPanels.forEach(function (panel) {
        panel.hidden = panel.getAttribute('data-hm-year') !== String(year);
      });
    }

    yearTabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        activateYear(tab.getAttribute('data-hm-tab'));
      });
    });
    if (yearTabs.length) {
      activateYear(yearTabs[0].getAttribute('data-hm-tab'));
    }

    // Floating tooltip for heatmap cells
    function showTooltip(cell) {
      if (!tooltip) return;
      var date = cell.getAttribute('data-hm-date');
      var count = Number(cell.getAttribute('data-hm-count') || 0);
      var titles = cell.getAttribute('data-hm-titles') || '';

      tooltip.innerHTML = '';
      var head = document.createElement('div');
      head.className = 'ba-tt-date';
      head.textContent = count > 0 ? date + ' · ' + count + ' 篇' : date + ' · 无更新';
      tooltip.appendChild(head);

      if (titles) {
        var body = document.createElement('div');
        body.className = 'ba-tt-titles';
        body.textContent = titles;
        tooltip.appendChild(body);
      }

      tooltip.hidden = false;
      var rect = cell.getBoundingClientRect();
      var ttRect = tooltip.getBoundingClientRect();
      var left = Math.max(10, Math.min(window.innerWidth - ttRect.width - 10, rect.left + rect.width / 2 - ttRect.width / 2));
      var top = rect.top - ttRect.height - 8;
      if (top < 10) top = rect.bottom + 8;
      tooltip.style.left = left + 'px';
      tooltip.style.top = top + 'px';
    }

    function hideTooltip() {
      if (tooltip) tooltip.hidden = true;
    }

    hmCells.forEach(function (cell) {
      cell.addEventListener('mouseenter', function () {
        showTooltip(cell);
      });
      cell.addEventListener('mouseleave', hideTooltip);
      cell.addEventListener('focus', function () {
        showTooltip(cell);
      });
      cell.addEventListener('blur', hideTooltip);

      if (cell.classList.contains('is-clickable')) {
        cell.addEventListener('click', function () {
          var clickedDate = cell.getAttribute('data-hm-date');
          state.date = state.date === clickedDate ? '' : clickedDate;
          state.month = '';
          if (state.date) {
            state.kind = 'all';
          }
          applyAll();
          if (state.date && tableSection) {
            tableSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        });
      }
    });

    // Category distribution bars -> filter table
    catBars.forEach(function (bar) {
      bar.addEventListener('click', function () {
        var slug = bar.getAttribute('data-bar-cat');
        var barKind = bar.getAttribute('data-bar-kind');
        if (state.cat === slug) {
          state.cat = 'all';
        } else {
          state.cat = slug;
          state.kind = barKind === 'note' ? 'note' : 'paper';
        }
        applyAll();
        if (tableSection) {
          tableSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });

    // Monthly bar chart -> filter table by month
    monthBars.forEach(function (col) {
      col.addEventListener('click', function () {
        var mKey = col.getAttribute('data-bar-month');
        state.month = state.month === mKey ? '' : mKey;
        state.date = '';
        if (state.month) {
          state.kind = 'all';
        }
        applyAll();
        if (state.month && tableSection) {
          tableSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });

    // Kind tabs
    kindTabs.forEach(function (btn) {
      btn.addEventListener('click', function () {
        state.kind = btn.getAttribute('data-kind-tab');
        if (state.kind === 'note' && state.cat !== 'all' && state.cat !== 'tech-notes') {
          state.cat = 'all';
        }
        if (state.kind === 'paper' && state.cat === 'tech-notes') {
          state.cat = 'all';
        }
        applyAll();
      });
    });

    // Org pills
    orgPills.forEach(function (pill) {
      pill.addEventListener('click', function () {
        var label = pill.getAttribute('data-org-filter');
        state.org = state.org === label ? '' : label;
        if (state.org && state.kind === 'note') {
          state.kind = 'paper';
        }
        applyAll();
      });
    });

    // Clickable category chips inside table rows
    rowCatChips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        var slug = chip.getAttribute('data-row-cat');
        state.cat = state.cat === slug ? 'all' : slug;
        if (slug === 'tech-notes') state.kind = 'note';
        else if (state.cat !== 'all') state.kind = 'paper';
        applyAll();
      });
    });

    // Search & selects
    if (searchInput) {
      searchInput.addEventListener('input', function () {
        state.query = searchInput.value.trim().toLowerCase();
        applyAll();
      });
    }

    if (searchClear) {
      searchClear.addEventListener('click', function () {
        state.query = '';
        if (searchInput) {
          searchInput.value = '';
          searchInput.focus();
        }
        applyAll();
      });
    }

    if (catSelect) {
      catSelect.addEventListener('change', function () {
        state.cat = catSelect.value;
        if (state.cat === 'tech-notes') state.kind = 'note';
        else if (state.cat !== 'all' && state.kind === 'note') state.kind = 'paper';
        applyAll();
      });
    }

    if (sortSelect) {
      sortSelect.addEventListener('change', function () {
        state.sort = sortSelect.value;
        applySort();
      });
    }

    function resetFilters() {
      state.kind = 'paper';
      state.cat = 'all';
      state.org = '';
      state.date = '';
      state.month = '';
      state.query = '';
      if (searchInput) searchInput.value = '';
      applyAll();
    }

    if (resetBtn) resetBtn.addEventListener('click', resetFilters);
    if (emptyResetBtn) emptyResetBtn.addEventListener('click', resetFilters);

    function applySort() {
      var sorted = rows.slice().sort(function (a, b) {
        if (state.sort === 'date-asc') {
          return a.getAttribute('data-date').localeCompare(b.getAttribute('data-date'));
        }
        if (state.sort === 'title') {
          return a.getAttribute('data-title').localeCompare(b.getAttribute('data-title'), 'zh-CN');
        }
        if (state.sort === 'cat') {
          var cCmp = a.getAttribute('data-cat').localeCompare(b.getAttribute('data-cat'));
          if (cCmp !== 0) return cCmp;
          return b.getAttribute('data-date').localeCompare(a.getAttribute('data-date'));
        }
        return b.getAttribute('data-date').localeCompare(a.getAttribute('data-date'));
      });
      var frag = document.createDocumentFragment();
      sorted.forEach(function (tr) {
        frag.appendChild(tr);
      });
      tbody.appendChild(frag);
    }

    function rowMatches(tr) {
      var kind = tr.getAttribute('data-kind');
      var cat = tr.getAttribute('data-cat');
      var date = tr.getAttribute('data-date');
      var month = tr.getAttribute('data-month');

      if (state.kind !== 'all' && kind !== state.kind) return false;
      if (state.cat !== 'all' && cat !== state.cat) return false;
      if (state.date && date !== state.date) return false;
      if (state.month && month !== state.month) return false;

      if (state.org) {
        var orgList = (tr.getAttribute('data-orgs') || '').split('|');
        if (orgList.indexOf(state.org) === -1) return false;
      }

      if (state.query) {
        var haystack = tr.getAttribute('data-search') || '';
        var tokens = state.query.split(/\s+/).filter(Boolean);
        for (var i = 0; i < tokens.length; i++) {
          if (haystack.indexOf(tokens[i]) === -1) return false;
        }
      }
      return true;
    }

    function renderActiveChips() {
      if (!filterTags) return;
      filterTags.innerHTML = '';

      function addTag(text, onRemove) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'ba-active-tag';
        btn.innerHTML = text + ' <span aria-hidden="true">×</span>';
        btn.addEventListener('click', onRemove);
        filterTags.appendChild(btn);
      }

      if (state.date) {
        addTag('日期：' + state.date, function () {
          state.date = '';
          applyAll();
        });
      }
      if (state.month) {
        addTag('月份：' + state.month, function () {
          state.month = '';
          applyAll();
        });
      }
      if (state.org) {
        addTag('机构：' + state.org, function () {
          state.org = '';
          applyAll();
        });
      }
      if (state.cat !== 'all' && catSelect) {
        var selectedOpt = catSelect.querySelector('option[value="' + state.cat + '"]');
        var catText = selectedOpt ? selectedOpt.textContent.replace(/\s*\(\d+\)$/, '') : state.cat;
        addTag('专题：' + catText, function () {
          state.cat = 'all';
          applyAll();
        });
      }
    }

    function applyAll() {
      var visible = 0;
      rows.forEach(function (tr) {
        var show = rowMatches(tr);
        tr.hidden = !show;
        if (show) visible += 1;
      });

      kindTabs.forEach(function (btn) {
        btn.classList.toggle('is-active', btn.getAttribute('data-kind-tab') === state.kind);
      });

      orgPills.forEach(function (pill) {
        pill.classList.toggle('is-active', pill.getAttribute('data-org-filter') === state.org);
      });

      catBars.forEach(function (bar) {
        bar.classList.toggle('is-active', bar.getAttribute('data-bar-cat') === state.cat);
      });

      monthBars.forEach(function (col) {
        col.classList.toggle('is-active', col.getAttribute('data-bar-month') === state.month);
      });

      hmCells.forEach(function (cell) {
        cell.classList.toggle('is-selected', Boolean(state.date) && cell.getAttribute('data-hm-date') === state.date);
      });

      if (catSelect) catSelect.value = state.cat;
      if (searchClear) searchClear.hidden = !state.query;
      if (resultCount) resultCount.innerHTML = '当前显示 <strong>' + visible + '</strong> 篇';

      var hasFilter =
        state.cat !== 'all' ||
        Boolean(state.org) ||
        Boolean(state.date) ||
        Boolean(state.month) ||
        Boolean(state.query);
      if (resetBtn) resetBtn.hidden = !hasFilter;
      if (emptyState) emptyState.hidden = visible > 0;

      renderActiveChips();
    }

    applySort();
    applyAll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBlogArchive);
  } else {
    initBlogArchive();
  }
})();
