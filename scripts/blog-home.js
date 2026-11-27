'use strict';

const { escapeHTML, stripHTML, url_for: urlFor } = require('hexo-util');

const PAPER_CATEGORIES = [
  { name: '智能体与Harness', slug: 'agents' },
  { name: '强化学习与后训练', slug: 'reinforcement-learning' },
  { name: '模型架构与推理效率', slug: 'architecture' },
  { name: '理论与数学推理', slug: 'theory' },
  { name: '评测与验证', slug: 'evaluation' },
  { name: '多模态与具身智能', slug: 'multimodal' },
  { name: '安全与可解释性', slug: 'safety' },
];

const TECH_CATEGORY = { name: '技术解析', slug: 'tech-notes' };

const PAPER_SLUG_MAP = new Map(PAPER_CATEGORIES.map((c) => [c.name, c.slug]));

function getPrimaryCategory(post) {
  if (!post.paper_reading) {
    const topCat = post.categories && post.categories.toArray().find((entry) => !entry.parent);
    if (topCat && topCat.name !== TECH_CATEGORY.name) {
      return { name: topCat.name, slug: TECH_CATEGORY.slug, kind: 'note' };
    }
    return { name: TECH_CATEGORY.name, slug: TECH_CATEGORY.slug, kind: 'note' };
  }

  const topCat = post.categories && post.categories.toArray().find((entry) => !entry.parent);
  const name = topCat ? topCat.name : '其他论文';
  const slug = PAPER_SLUG_MAP.get(name) || `paper-${encodeURIComponent(name)}`;
  return { name, slug, kind: 'paper' };
}

function getCleanTitle(post) {
  return String(post.title || '').replace(/^Paper精读[｜|]\s*/u, '').trim();
}

function getExcerpt(post) {
  const raw = post.description || post.excerpt || '';
  if (raw) {
    return stripHTML(String(raw)).replace(/\s+/g, ' ').trim();
  }
  if (post.content) {
    return stripHTML(String(post.content)).replace(/\s+/g, ' ').trim().slice(0, 160);
  }
  return '';
}

function estimateReadMinutes(post) {
  const text = stripHTML(String(post.content || post._content || ''));
  const zhCount = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const enWords = (text.replace(/[\u4e00-\u9fa5]/g, ' ').match(/[A-Za-z0-9_]+/g) || []).length;
  const mins = Math.max(1, Math.round(zhCount / 380 + enWords / 220));
  return mins;
}

function getDisplayTags(post) {
  if (!post.tags) return [];
  return post.tags
    .toArray()
    .map((t) => t.name)
    .filter((name) => name && name !== 'Paper精读')
    .slice(0, 3);
}

function renderCard(item, { isPinned = false } = {}) {
  const kindLabel = item.kind === 'paper' ? '论文精读' : '技术解析';
  const kindClass = item.kind === 'paper' ? 'is-paper' : 'is-note';
  const topicBadge = item.kind === 'paper'
    ? `<button type="button" class="bh-chip bh-chip-topic" data-cat-jump="${escapeHTML(item.catSlug)}" title="筛选「${escapeHTML(item.catName)}」">${escapeHTML(item.catName)}</button>`
    : '';
  const pinBadge = isPinned
    ? `<span class="bh-badge bh-badge-pin">置顶推荐</span>`
    : '';
  const tagsHtml = item.tags.length
    ? `<div class="bh-card-tags">${item.tags.map((t) => `<span class="bh-tag">#${escapeHTML(t)}</span>`).join('')}</div>`
    : '';

  const searchBlob = escapeHTML(
    `${item.title} ${item.excerpt} ${item.catName} ${item.tags.join(' ')} ${item.monthKey}`.toLowerCase()
  );

  if (isPinned) {
    return `<article class="bh-pinned-card" data-kind="${item.kind}" data-cat="${escapeHTML(item.catSlug)}">
      <div class="bh-card-top">
        <div class="bh-card-badges">
          ${pinBadge}
          <span class="bh-badge ${kindClass}">${kindLabel}</span>
          ${topicBadge}
        </div>
        <div class="bh-card-date">
          <time datetime="${escapeHTML(item.isoDate)}">${escapeHTML(item.fullDate)}</time>
          <span class="bh-dot" aria-hidden="true">·</span>
          <span>约 ${item.readMinutes} 分钟</span>
        </div>
      </div>
      <h2 class="bh-pinned-title"><a href="${escapeHTML(item.url)}">${escapeHTML(item.title)}</a></h2>
      ${item.excerpt ? `<p class="bh-card-excerpt">${escapeHTML(item.excerpt)}</p>` : ''}
      <div class="bh-pinned-footer">
        ${tagsHtml}
        <a class="bh-read-link" href="${escapeHTML(item.url)}">开始阅读 →</a>
      </div>
    </article>`;
  }

  return `<article class="bh-card" data-kind="${item.kind}" data-cat="${escapeHTML(item.catSlug)}" data-month="${escapeHTML(item.monthKey)}" data-search="${searchBlob}">
    <div class="bh-card-top">
      <div class="bh-card-badges">
        <span class="bh-badge ${kindClass}">${kindLabel}</span>
        ${topicBadge}
      </div>
      <div class="bh-card-date">
        <time datetime="${escapeHTML(item.isoDate)}" title="${escapeHTML(item.fullDate)}">${escapeHTML(item.shortDate)}</time>
        <span class="bh-dot" aria-hidden="true">·</span>
        <span>${item.readMinutes} min</span>
      </div>
    </div>
    <h3 class="bh-card-title">
      <a href="${escapeHTML(item.url)}">${escapeHTML(item.title)}</a>
    </h3>
    ${item.excerpt ? `<p class="bh-card-excerpt">${escapeHTML(item.excerpt)}</p>` : ''}
    ${tagsHtml}
  </article>`;
}

function buildHomepageHtml(locals, context) {
  const rawPosts = locals.posts
    .filter((p) => !p.hide)
    .sort('-date')
    .toArray();

  const items = rawPosts.map((post) => {
    const cat = getPrimaryCategory(post);
    const year = post.date.format('YYYY');
    const month = post.date.format('MM');
    return {
      title: getCleanTitle(post),
      url: urlFor.call(context, post.path),
      excerpt: getExcerpt(post),
      isoDate: post.date.format(),
      fullDate: post.date.format('YYYY-MM-DD'),
      shortDate: post.date.format('MM-DD'),
      year,
      month,
      monthKey: `${year}-${month}`,
      monthLabel: `${year} 年 ${parseInt(month, 10)} 月`,
      catName: cat.name,
      catSlug: cat.slug,
      kind: cat.kind,
      sticky: Number(post.sticky) || 0,
      tags: getDisplayTags(post),
      readMinutes: estimateReadMinutes(post),
    };
  });

  const totalCount = items.length;
  const noteItems = items.filter((i) => i.kind === 'note');
  const paperItems = items.filter((i) => i.kind === 'paper');
  const pinnedItems = items
    .filter((i) => i.sticky > 0)
    .sort((a, b) => b.sticky - a.sticky);
  const latestDate = items[0] ? items[0].fullDate : '';

  // Count papers per direction in canonical order (+ any unexpected ones)
  const paperCatCounts = new Map(PAPER_CATEGORIES.map((c) => [c.slug, { ...c, count: 0 }]));
  for (const item of paperItems) {
    if (!paperCatCounts.has(item.catSlug)) {
      paperCatCounts.set(item.catSlug, { name: item.catName, slug: item.catSlug, count: 0 });
    }
    paperCatCounts.get(item.catSlug).count += 1;
  }
  const activePaperCats = [...paperCatCounts.values()].filter((c) => c.count > 0);

  // Group all posts chronologically by Year-Month
  const monthMap = new Map();
  const yearMap = new Map();
  for (const item of items) {
    if (!monthMap.has(item.monthKey)) {
      monthMap.set(item.monthKey, {
        key: item.monthKey,
        year: item.year,
        month: item.month,
        label: item.monthLabel,
        items: [],
      });
    }
    monthMap.get(item.monthKey).items.push(item);

    if (!yearMap.has(item.year)) {
      yearMap.set(item.year, []);
    }
    const yearMonths = yearMap.get(item.year);
    if (!yearMonths.includes(item.monthKey)) {
      yearMonths.push(item.monthKey);
    }
  }
  const monthGroups = [...monthMap.values()];

  // Sidebar: Category Nav
  const paperCatLinks = activePaperCats
    .map(
      (cat) => `<li>
        <button type="button" class="bh-nav-btn" data-filter-cat="${escapeHTML(cat.slug)}">
          <span class="bh-nav-label"><span class="bh-nav-dot is-paper"></span>${escapeHTML(cat.name)}</span>
          <span class="bh-nav-count">${cat.count}</span>
        </button>
      </li>`
    )
    .join('\n');

  // Sidebar: Timeline Nav grouped by year
  const timelineYearsHtml = [...yearMap.entries()]
    .map(([year, monthKeys]) => {
      const monthLinks = monthKeys
        .map((mKey) => {
          const mg = monthMap.get(mKey);
          const monthNum = parseInt(mg.month, 10);
          return `<li data-timeline-item="${escapeHTML(mKey)}">
            <a class="bh-time-link" href="#month-${escapeHTML(mKey)}" data-month-target="${escapeHTML(mKey)}">
              <span>${monthNum} 月</span>
              <span class="bh-nav-count" data-month-count="${escapeHTML(mKey)}">${mg.items.length}</span>
            </a>
          </li>`;
        })
        .join('\n');

      return `<div class="bh-time-year" data-timeline-year="${escapeHTML(year)}">
        <div class="bh-time-year-label">${escapeHTML(year)}</div>
        <ul class="bh-time-months">${monthLinks}</ul>
      </div>`;
    })
    .join('\n');

  // Mobile quick filter bar chips
  const mobileChips = [
    { slug: 'all', label: `全部 (${totalCount})`, kind: 'all' },
    { slug: TECH_CATEGORY.slug, label: `技术解析 (${noteItems.length})`, kind: 'note' },
    { slug: 'papers', label: `全部论文 (${paperItems.length})`, kind: 'paper' },
    ...activePaperCats.map((c) => ({ slug: c.slug, label: `${c.name} (${c.count})`, kind: 'paper' })),
  ]
    .map(
      (chip, idx) =>
        `<button type="button" class="bh-mobile-chip${idx === 0 ? ' is-active' : ''}" data-filter-cat="${escapeHTML(chip.slug)}">${escapeHTML(chip.label)}</button>`
    )
    .join('\n');

  // Pinned section
  const pinnedHtml = pinnedItems.length
    ? `<section class="bh-pinned-section" id="bh-pinned-section" aria-label="置顶推荐">
        <div class="bh-section-kicker">
          <span>精选置顶</span>
          <span class="bh-kicker-note">系统梳理 · 推荐从这里开始</span>
        </div>
        <div class="bh-pinned-grid">
          ${pinnedItems.map((item) => renderCard(item, { isPinned: true })).join('\n')}
        </div>
      </section>`
    : '';

  // Month timeline sections
  const feedHtml = monthGroups
    .map(
      (mg) => `<section class="bh-month-group" id="month-${escapeHTML(mg.key)}" data-month-group="${escapeHTML(mg.key)}" data-year="${escapeHTML(mg.year)}">
        <header class="bh-month-header">
          <span class="bh-month-dot" aria-hidden="true"></span>
          <h2 class="bh-month-title">${escapeHTML(mg.label)}</h2>
          <span class="bh-month-count" data-group-count="${escapeHTML(mg.key)}">${mg.items.length} 篇</span>
        </header>
        <div class="bh-cards">
          ${mg.items.map((item) => renderCard(item)).join('\n')}
        </div>
      </section>`
    )
    .join('\n');

  return `<div class="blog-home" id="blog-home">
    <header class="bh-topbar">
      <div class="bh-stats" aria-label="站点内容概览">
        <span class="bh-stat-pill"><strong>${totalCount}</strong> 篇文章</span>
        <span class="bh-stat-sep" aria-hidden="true">/</span>
        <span class="bh-stat-pill is-note">技术解析 <strong>${noteItems.length}</strong></span>
        <span class="bh-stat-sep" aria-hidden="true">/</span>
        <span class="bh-stat-pill is-paper">论文精读 <strong>${paperItems.length}</strong>（${activePaperCats.length} 方向）</span>
        ${latestDate ? `<span class="bh-stat-sep bh-hide-sm" aria-hidden="true">·</span><span class="bh-stat-updated bh-hide-sm">更新于 ${escapeHTML(latestDate)}</span>` : ''}
      </div>
      <div class="bh-controls">
        <label class="bh-search-box" aria-label="快速筛选文章">
          <i class="iconfont icon-search" aria-hidden="true"></i>
          <input id="bh-filter-input" type="search" placeholder="筛选标题、摘要或标签… (按 / )" autocomplete="off" />
          <button type="button" id="bh-filter-clear" class="bh-clear-btn" aria-label="清空筛选" hidden>×</button>
        </label>
        <div class="bh-view-toggle" role="group" aria-label="切换阅读视图">
          <button type="button" class="bh-view-btn is-active" data-view="card" title="卡片视图（含摘要）">卡片</button>
          <button type="button" class="bh-view-btn" data-view="compact" title="紧凑清单视图">清单</button>
        </div>
      </div>
    </header>

    <div class="bh-mobile-bar" aria-label="移动端分类筛选">
      <div class="bh-mobile-chips">${mobileChips}</div>
    </div>

    <div class="bh-workspace">
      <aside class="bh-sidebar" aria-label="分类与时间线导航">
        <nav class="bh-side-block" aria-label="内容分类">
          <div class="bh-side-heading">内容导航</div>
          <ul class="bh-nav-list">
            <li>
              <button type="button" class="bh-nav-btn is-active" data-filter-cat="all">
                <span class="bh-nav-label">全部文章</span>
                <span class="bh-nav-count">${totalCount}</span>
              </button>
            </li>
          </ul>

          <div class="bh-side-subheading">基础专栏</div>
          <ul class="bh-nav-list">
            <li>
              <button type="button" class="bh-nav-btn" data-filter-cat="${escapeHTML(TECH_CATEGORY.slug)}">
                <span class="bh-nav-label"><span class="bh-nav-dot is-note"></span>${escapeHTML(TECH_CATEGORY.name)}</span>
                <span class="bh-nav-count">${noteItems.length}</span>
              </button>
            </li>
          </ul>

          <div class="bh-side-subheading">论文精读 · 主题方向</div>
          <ul class="bh-nav-list">
            <li>
              <button type="button" class="bh-nav-btn" data-filter-cat="papers">
                <span class="bh-nav-label"><span class="bh-nav-dot is-paper-all"></span>全部论文精读</span>
                <span class="bh-nav-count">${paperItems.length}</span>
              </button>
            </li>
            ${paperCatLinks}
          </ul>
        </nav>

        <nav class="bh-side-block bh-timeline-block" aria-label="时间线导航">
          <div class="bh-side-heading">时间线导航</div>
          <div class="bh-timeline-nav" id="bh-timeline-nav">
            ${timelineYearsHtml}
          </div>
        </nav>
      </aside>

      <div class="bh-main">
        ${pinnedHtml}

        <div class="bh-active-bar" id="bh-active-bar">
          <div class="bh-active-info">
            <span class="bh-active-label" id="bh-active-label">全部文章</span>
            <span class="bh-active-count" id="bh-active-count">${totalCount} 篇</span>
          </div>
          <button type="button" class="bh-reset-btn" id="bh-reset-btn" hidden>查看全部</button>
        </div>

        <div class="bh-feed" id="bh-feed">
          ${feedHtml}
        </div>

        <div class="bh-empty" id="bh-empty" hidden>
          <p>没有匹配的文章，试试切换分类或换个关键词。</p>
          <button type="button" class="bh-reset-btn" id="bh-empty-reset">重置筛选</button>
        </div>
      </div>
    </div>
  </div>`;
}

// Register after Fluid's pages.js so this cleanly replaces the default index generator.
hexo.on('generateBefore', function () {
  hexo.extend.generator.register('index', function (locals) {
    const theme = this.theme.config || {};
    const slogan = (theme.index && theme.index.slogan && theme.index.slogan.enable && theme.index.slogan.text)
      || this.config.subtitle
      || this.config.title;

    return {
      path: 'index.html',
      layout: 'page',
      data: {
        __index: true,
        title: slogan,
        subtitle: slogan,
        banner_img: (theme.index && theme.index.banner_img) || '/img/bg.png',
        banner_img_height: (theme.index && theme.index.banner_img_height) || 30,
        banner_mask_alpha: (theme.index && theme.index.banner_mask_alpha) ?? 0.42,
        comments: false,
        content: buildHomepageHtml(locals, this),
      },
    };
  });
});
