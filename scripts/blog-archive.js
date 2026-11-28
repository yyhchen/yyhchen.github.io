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

// Canonical major lab / university buckets for quick-filter chips.
const CANONICAL_ORGS = [
  { label: 'DeepMind / Google', patterns: [/deepmind/i, /google/i] },
  { label: '清华大学', patterns: [/清华/, /tsinghua/i] },
  { label: '北京大学', patterns: [/北大/, /北京大/, /peking/i, /\bpku\b/i] },
  { label: 'DeepSeek', patterns: [/deepseek/i] },
  { label: 'MIT', patterns: [/\bmit\b/i, /massachusetts institute/i] },
  { label: 'Stanford', patterns: [/stanford/i] },
  { label: 'OpenAI', patterns: [/openai/i] },
  { label: 'Anthropic', patterns: [/anthropic/i] },
  { label: 'NVIDIA', patterns: [/nvidia/i] },
  { label: 'Qwen / Alibaba', patterns: [/qwen/i, /alibaba/i, /阿里/] },
  { label: 'Microsoft', patterns: [/microsoft/i, /微软/] },
  { label: 'UC Berkeley', patterns: [/berkeley/i, /\bucb\b/i] },
  { label: 'ByteDance / Seed', patterns: [/bytedance/i, /字节/, /\bseed\b/i] },
  { label: '小红书', patterns: [/小红书/, /xiaohongshu/i] },
  { label: '上海交大', patterns: [/上海交通/, /\bsjtu\b/i] },
];

function cleanMdText(str) {
  return String(str || '')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
    .replace(/\*\*/g, '')
    .replace(/`/g, '')
    .replace(/^\s*[:：\-–—]+\s*/, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractCiteMeta(post) {
  const raw = String(post.raw || post._content || '');
  const citeMatch = raw.match(/>\s*\[!cite\][^\n]*\n((?:>.*\n?)+)/);
  const block = citeMatch ? citeMatch[1] : '';

  // Primary URL (prefer arXiv, then alphaXiv, then any http URL inside cite block)
  let paperUrl = '';
  let linkLabel = '';
  const arxivUrlMatch = block.match(/https?:\/\/arxiv\.org\/(?:abs|pdf)\/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)/i);
  if (arxivUrlMatch) {
    paperUrl = `https://arxiv.org/abs/${arxivUrlMatch[1].replace(/v\d+$/i, '')}`;
    linkLabel = `arXiv:${arxivUrlMatch[1].replace(/v\d+$/i, '')}`;
  } else {
    const arxivIdLine = block.match(/\*\*arXiv[：:]?\*\*[：:]?\s*\*?\*?\s*([0-9]{4}\.[0-9]{4,5})/i);
    if (arxivIdLine) {
      paperUrl = `https://arxiv.org/abs/${arxivIdLine[1]}`;
      linkLabel = `arXiv:${arxivIdLine[1]}`;
    } else {
      const alphaMatch = block.match(/https?:\/\/(?:www\.)?alphaxiv\.org\/abs\/[^\s)\]>]+/i);
      const anyUrlMatch = alphaMatch || block.match(/https?:\/\/[^\s)\]>]+/i);
      if (anyUrlMatch) {
        paperUrl = anyUrlMatch[0].replace(/[.,;]+$/, '');
        if (/alphaxiv\.org/i.test(paperUrl)) linkLabel = 'alphaXiv';
        else if (/github\.com/i.test(paperUrl)) linkLabel = 'GitHub';
        else if (/openai\.com/i.test(paperUrl)) linkLabel = 'OpenAI';
        else if (/anthropic\.com/i.test(paperUrl)) linkLabel = 'Anthropic';
        else if (/huggingface\.co/i.test(paperUrl)) linkLabel = 'HuggingFace';
        else linkLabel = '原文';
      }
    }
  }

  // Organization / Institution
  let org = '';
  const orgLine =
    block.match(/\*\*机构[：:]?\*\*[：:]?\s*([^\n]+)/) ||
    block.match(/\*\*来源[：:]?\*\*[：:]?\s*([^\n]+)/) ||
    block.match(/\*\*团队[：:]?\*\*[：:]?\s*([^\n]+)/) ||
    block.match(/\*\*单位[：:]?\*\*[：:]?\s*([^\n]+)/) ||
    block.match(/(?:^|\n)>\s*[-*]?\s*(?:机构|来源|单位)[：:]\s*([^\n]+)/);

  if (orgLine) {
    const candidate = cleanMdText(orgLine[1]);
    if (!/^https?:\/\//i.test(candidate)) {
      org = candidate;
    }
  }

  // Fallback 1: parenthetical affiliation at end of Author line, e.g. "Kaiming He (MIT)"
  if (!org) {
    const authorLine = block.match(/\*\*作者[：:]?\*\*[：:]?\s*([^\n]+)/);
    if (authorLine) {
      const cleaned = cleanMdText(authorLine[1]);
      const parenMatches = [...cleaned.matchAll(/[（(]([^()（）]{2,45})[)）]/g)];
      if (parenMatches.length) {
        org = parenMatches[parenMatches.length - 1][1].trim();
      }
    }
  }

  // Fallback 2: infer from URL/title
  if (!org) {
    const hint = `${paperUrl} ${block} ${post.title || ''}`;
    if (/deepseek/i.test(hint)) org = 'DeepSeek-AI';
    else if (/openai\.com/i.test(hint)) org = 'OpenAI';
    else if (/anthropic/i.test(hint)) org = 'Anthropic';
    else if (/google-research|deepmind/i.test(hint)) org = 'Google Research';
    else if (/qwen/i.test(hint)) org = 'Qwen Team';
    else if (/glm/i.test(hint)) org = 'Z.ai / Zhipu';
    else if (/nvidia/i.test(hint)) org = 'NVIDIA';
  }

  // Keep org display tidy (preserve full text in title attribute)
  const orgFull = org;
  const orgShort = org.length > 42 ? `${org.slice(0, 40)}…` : org;

  // Match canonical org tags
  const canonicalOrgs = [];
  const matchSource = `${orgFull} ${paperUrl}`;
  for (const co of CANONICAL_ORGS) {
    if (co.patterns.some((re) => re.test(matchSource))) {
      canonicalOrgs.push(co.label);
    }
  }

  return { paperUrl, linkLabel, orgFull, orgShort, canonicalOrgs };
}

function getPrimaryCategory(post) {
  if (!post.paper_reading) {
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
  if (raw) return stripHTML(String(raw)).replace(/\s+/g, ' ').trim();
  if (post.content) return stripHTML(String(post.content)).replace(/\s+/g, ' ').trim().slice(0, 150);
  return '';
}

function countWords(post) {
  const text = stripHTML(String(post.content || post._content || ''));
  const zh = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const en = (text.replace(/[\u4e00-\u9fa5]/g, ' ').match(/[A-Za-z0-9_]+/g) || []).length;
  return zh + en;
}

function formatDateKey(y, m, d) {
  return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
}

function buildHeatmapYear(year, dayPostsMap) {
  const start = new Date(Date.UTC(year, 0, 1));
  const end = new Date(Date.UTC(year, 11, 31));
  // Align to Monday (0 = Mon ... 6 = Sun)
  const startDow = (start.getUTCDay() + 6) % 7;
  const gridStart = new Date(start.getTime() - startDow * 86400000);

  const cells = [];
  const monthLabels = new Map();
  let cur = new Date(gridStart.getTime());
  let weekIndex = 0;
  let yearPostTotal = 0;
  let yearActiveDays = 0;

  while (cur <= end || ((cur.getUTCDay() + 6) % 7) !== 0) {
    const dow = (cur.getUTCDay() + 6) % 7;
    if (dow === 0 && cur > gridStart) weekIndex += 1;

    const cy = cur.getUTCFullYear();
    const cm = cur.getUTCMonth() + 1;
    const cd = cur.getUTCDate();
    const inYear = cy === year;

    if (inYear && cd === 1 && !monthLabels.has(cm)) {
      monthLabels.set(cm, weekIndex);
    }

    if (!inYear) {
      cells.push('<span class="ba-hm-cell is-out" aria-hidden="true"></span>');
    } else {
      const key = formatDateKey(cy, cm, cd);
      const dayPosts = dayPostsMap.get(key) || [];
      const count = dayPosts.length;
      if (count > 0) {
        yearPostTotal += count;
        yearActiveDays += 1;
      }
      const level = count === 0 ? 0 : count === 1 ? 1 : count === 2 ? 2 : count <= 4 ? 3 : 4;
      const titles = dayPosts.map((p) => p.title).join(' ｜ ');
      const label = count > 0 ? `${key} · ${count} 篇：${titles}` : `${key} · 无更新`;

      if (count > 0) {
        cells.push(
          `<button type="button" class="ba-hm-cell lvl-${level} is-clickable" data-hm-date="${key}" data-hm-count="${count}" data-hm-titles="${escapeHTML(titles)}" title="${escapeHTML(label)}" aria-label="${escapeHTML(label)}"></button>`
        );
      } else {
        cells.push(
          `<span class="ba-hm-cell lvl-0" data-hm-date="${key}" data-hm-count="0" title="${escapeHTML(label)}"></span>`
        );
      }
    }

    cur = new Date(cur.getTime() + 86400000);
  }

  const totalWeeks = weekIndex + 1;
  const monthHeaderCells = [];
  for (let w = 0; w < totalWeeks; w++) {
    let text = '';
    for (const [m, mw] of monthLabels.entries()) {
      if (mw === w) {
        text = `${m}月`;
        break;
      }
    }
    monthHeaderCells.push(`<span class="ba-hm-month">${text}</span>`);
  }

  return {
    year,
    yearPostTotal,
    yearActiveDays,
    html: `<div class="ba-hm-panel" data-hm-year="${year}">
      <div class="ba-hm-summary">
        <strong>${year} 年</strong>共更新 <strong>${yearPostTotal}</strong> 篇，活跃 <strong>${yearActiveDays}</strong> 天
        <span class="ba-hm-hint">（点击有颜色的日期方块可筛选当天文章）</span>
      </div>
      <div class="ba-hm-scroll">
        <div class="ba-hm-grid-wrap" style="--ba-weeks: ${totalWeeks}">
          <div class="ba-hm-months">${monthHeaderCells.join('')}</div>
          <div class="ba-hm-body">
            <div class="ba-hm-weekdays" aria-hidden="true">
              <span>一</span><span></span><span>三</span><span></span><span>五</span><span></span><span>日</span>
            </div>
            <div class="ba-hm-cells">${cells.join('')}</div>
          </div>
        </div>
      </div>
    </div>`,
  };
}

function buildArchiveDashboardHtml(locals, context) {
  const rawPosts = locals.posts.filter((p) => !p.hide).sort('-date').toArray();

  let totalWords = 0;
  const dayPostsMap = new Map();
  const monthCounts = new Map();
  const catCounts = new Map([
    [TECH_CATEGORY.slug, { name: TECH_CATEGORY.name, slug: TECH_CATEGORY.slug, kind: 'note', count: 0 }],
    ...PAPER_CATEGORIES.map((c) => [c.slug, { name: c.name, slug: c.slug, kind: 'paper', count: 0 }]),
  ]);
  const canonicalOrgCounts = new Map();
  const yearsSet = new Set();

  const items = rawPosts.map((post) => {
    const cat = getPrimaryCategory(post);
    const title = getCleanTitle(post);
    const excerpt = getExcerpt(post);
    const fullDate = post.date.format('YYYY-MM-DD');
    const monthKey = post.date.format('YYYY-MM');
    const year = post.date.year();
    const words = countWords(post);
    const cite = post.paper_reading ? extractCiteMeta(post) : { paperUrl: '', linkLabel: '', orgFull: '个人技术解析', orgShort: '个人技术解析', canonicalOrgs: [] };

    totalWords += words;
    yearsSet.add(year);

    if (!dayPostsMap.has(fullDate)) dayPostsMap.set(fullDate, []);
    dayPostsMap.get(fullDate).push({ title, url: urlFor.call(context, post.path) });

    monthCounts.set(monthKey, (monthCounts.get(monthKey) || 0) + 1);

    if (!catCounts.has(cat.slug)) {
      catCounts.set(cat.slug, { name: cat.name, slug: cat.slug, kind: cat.kind, count: 0 });
    }
    catCounts.get(cat.slug).count += 1;

    for (const co of cite.canonicalOrgs) {
      canonicalOrgCounts.set(co, (canonicalOrgCounts.get(co) || 0) + 1);
    }

    return {
      title,
      url: urlFor.call(context, post.path),
      excerpt,
      fullDate,
      isoDate: post.date.format(),
      monthKey,
      year,
      catName: cat.name,
      catSlug: cat.slug,
      kind: cat.kind,
      words,
      ...cite,
    };
  });

  const totalCount = items.length;
  const paperCount = items.filter((i) => i.kind === 'paper').length;
  const noteCount = items.filter((i) => i.kind === 'note').length;
  const activeDays = dayPostsMap.size;
  const sortedYears = [...yearsSet].sort((a, b) => b - a);
  const firstMonth = items.length ? items[items.length - 1].monthKey : '';
  const lastMonth = items.length ? items[0].monthKey : '';

  // Peak month
  let peakMonth = '';
  let peakMonthCount = 0;
  for (const [m, c] of monthCounts.entries()) {
    if (c > peakMonthCount) {
      peakMonthCount = c;
      peakMonth = m;
    }
  }

  // Word count display
  const wordsDisplay = totalWords >= 10000
    ? `${(totalWords / 10000).toFixed(1)} 万字`
    : `${totalWords} 字`;

  // Heatmap panels by year
  const heatmapYearPanels = sortedYears.map((y) => buildHeatmapYear(y, dayPostsMap));
  const yearOptionsHtml = heatmapYearPanels
    .map(
      (p, idx) =>
        `<option value="${p.year}"${idx === 0 ? ' selected' : ''}>${p.year} 年 · ${p.yearPostTotal} 篇</option>`
    )
    .join('\n');

  // Category distribution bars (sorted by count desc)
  const catList = [...catCounts.values()].filter((c) => c.count > 0).sort((a, b) => b.count - a.count);
  const maxCatCount = catList[0] ? catList[0].count : 1;
  const catBarsHtml = catList
    .map((c) => {
      const pct = Math.round((c.count / maxCatCount) * 100);
      const share = ((c.count / Math.max(1, totalCount)) * 100).toFixed(0);
      return `<button type="button" class="ba-cat-bar-row" data-bar-cat="${escapeHTML(c.slug)}" data-bar-kind="${c.kind}" title="点击筛选「${escapeHTML(c.name)}」">
        <span class="ba-cat-bar-name">${escapeHTML(c.name)}</span>
        <span class="ba-cat-bar-track">
          <span class="ba-cat-bar-fill ${c.kind === 'note' ? 'is-note' : 'is-paper'}" style="width: ${pct}%"></span>
        </span>
        <span class="ba-cat-bar-num">${c.count} <em>(${share}%)</em></span>
      </button>`;
    })
    .join('\n');

  // Monthly bar chart (up to 12 most recent active months, chronological left -> right)
  const recentMonths = [...monthCounts.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-12);
  const maxMonthVal = Math.max(1, ...recentMonths.map(([, c]) => c));
  const monthBarsHtml = recentMonths
    .map(([mKey, c]) => {
      const hPct = Math.max(10, Math.round((c / maxMonthVal) * 100));
      const shortLabel = mKey.slice(2); // e.g. 26-11
      return `<button type="button" class="ba-mbar-col" data-bar-month="${escapeHTML(mKey)}" title="${escapeHTML(mKey)}：${c} 篇（点击筛选）">
        <span class="ba-mbar-val">${c}</span>
        <span class="ba-mbar-track"><span class="ba-mbar-fill" style="height: ${hPct}%"></span></span>
        <span class="ba-mbar-lbl">${escapeHTML(shortLabel)}</span>
      </button>`;
    })
    .join('\n');

  // Top canonical org quick-filter pills (orgs with >= 2 papers, top 10)
  const topOrgs = [...canonicalOrgCounts.entries()]
    .filter(([, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  const orgPillsHtml = topOrgs.length
    ? `<div class="ba-org-pills" aria-label="按热门机构快速筛选">
        <span class="ba-org-pills-label">热门机构</span>
        ${topOrgs
          .map(
            ([label, count]) =>
              `<button type="button" class="ba-org-pill" data-org-filter="${escapeHTML(label)}">${escapeHTML(label)} <span>${count}</span></button>`
          )
          .join('\n')}
      </div>`
    : '';

  // Topic select options
  const topicOptionsHtml = [
    '<option value="all">全部专题方向</option>',
    ...PAPER_CATEGORIES.map((c) => {
      const cnt = (catCounts.get(c.slug) || {}).count || 0;
      return `<option value="${escapeHTML(c.slug)}">${escapeHTML(c.name)} (${cnt})</option>`;
    }),
    `<option value="${TECH_CATEGORY.slug}">${TECH_CATEGORY.name} (${noteCount})</option>`,
  ].join('\n');

  // Table rows
  const rowsHtml = items
    .map((item) => {
      const orgsData = item.canonicalOrgs.join('|');
      const searchBlob = `${item.title} ${item.excerpt} ${item.orgFull} ${item.catName} ${item.linkLabel} ${item.fullDate}`.toLowerCase();
      const orgCell = item.orgShort
        ? `<span class="ba-org-text" title="${escapeHTML(item.orgFull)}">${escapeHTML(item.orgShort)}</span>`
        : '<span class="ba-muted-dash">—</span>';
      const linkCell = item.paperUrl
        ? `<a class="ba-ext-link" href="${escapeHTML(item.paperUrl)}" target="_blank" rel="noopener noreferrer">${escapeHTML(item.linkLabel || '原文')} ↗</a>`
        : '<span class="ba-muted-dash">—</span>';
      const kindPill = item.kind === 'note'
        ? '<span class="ba-kind-dot is-note" title="技术解析"></span>'
        : '<span class="ba-kind-dot is-paper" title="论文精读"></span>';

      return `<tr class="ba-row"
        data-kind="${item.kind}"
        data-cat="${escapeHTML(item.catSlug)}"
        data-date="${escapeHTML(item.fullDate)}"
        data-month="${escapeHTML(item.monthKey)}"
        data-title="${escapeHTML(item.title.toLowerCase())}"
        data-orgs="${escapeHTML(orgsData)}"
        data-search="${escapeHTML(searchBlob)}">
        <td class="ba-col-date" data-label="日期">
          <time datetime="${escapeHTML(item.isoDate)}">${escapeHTML(item.fullDate)}</time>
        </td>
        <td class="ba-col-title" data-label="文章">
          <div class="ba-row-title-line">
            ${kindPill}
            <a class="ba-row-link" href="${escapeHTML(item.url)}">${escapeHTML(item.title)}</a>
          </div>
          ${item.excerpt ? `<p class="ba-row-excerpt">${escapeHTML(item.excerpt)}</p>` : ''}
        </td>
        <td class="ba-col-org" data-label="机构/来源">${orgCell}</td>
        <td class="ba-col-cat" data-label="专题">
          <button type="button" class="ba-cat-chip ${item.kind === 'note' ? 'is-note' : 'is-paper'}" data-row-cat="${escapeHTML(item.catSlug)}">${escapeHTML(item.catName)}</button>
        </td>
        <td class="ba-col-source" data-label="原文">${linkCell}</td>
      </tr>`;
    })
    .join('\n');

  return `<div class="blog-archive" id="blog-archive">
    <!-- Part 1: KPI Summary -->
    <section class="ba-kpis" aria-label="写作统计总览">
      <div class="ba-kpi">
        <div class="ba-kpi-label">累计文章</div>
        <div class="ba-kpi-value">${totalCount}</div>
        <div class="ba-kpi-sub">论文精读 ${paperCount} · 技术解析 ${noteCount}</div>
      </div>
      <div class="ba-kpi">
        <div class="ba-kpi-label">活跃写作日</div>
        <div class="ba-kpi-value">${activeDays}<span class="ba-kpi-unit">天</span></div>
        <div class="ba-kpi-sub">${escapeHTML(firstMonth)} ~ ${escapeHTML(lastMonth)}</div>
      </div>
      <div class="ba-kpi">
        <div class="ba-kpi-label">单月峰值</div>
        <div class="ba-kpi-value">${peakMonthCount}<span class="ba-kpi-unit">篇</span></div>
        <div class="ba-kpi-sub">最勤奋月份：${escapeHTML(peakMonth)}</div>
      </div>
      <div class="ba-kpi">
        <div class="ba-kpi-label">累计字数</div>
        <div class="ba-kpi-value">${escapeHTML(wordsDisplay)}</div>
        <div class="ba-kpi-sub">7 大论文专题 + 技术解析</div>
      </div>
    </section>

    <!-- Part 2: Heatmap + Analytics -->
    <section class="ba-analytics" aria-label="热力图与研究分布">
      <div class="ba-panel ba-heatmap-card">
        <div class="ba-panel-head">
          <div>
            <h2 class="ba-panel-title">更新热力图</h2>
            <p class="ba-panel-sub">整理土地，而非焦虑时光</p>
          </div>
          <select id="ba-year-select" class="ba-select ba-year-select" aria-label="选择年份">
            ${yearOptionsHtml}
          </select>
        </div>
        <div class="ba-hm-panels">
          ${heatmapYearPanels.map((p) => p.html).join('\n')}
        </div>
        <div class="ba-hm-legend">
          <span>少</span>
          <span class="ba-hm-cell lvl-0"></span>
          <span class="ba-hm-cell lvl-1"></span>
          <span class="ba-hm-cell lvl-2"></span>
          <span class="ba-hm-cell lvl-3"></span>
          <span class="ba-hm-cell lvl-4"></span>
          <span>多</span>
        </div>
      </div>

      <div class="ba-analytics-split">
        <div class="ba-panel">
          <div class="ba-panel-head">
            <h2 class="ba-panel-title">方向分布</h2>
            <span class="ba-panel-tip">点击筛选下方表格</span>
          </div>
          <div class="ba-cat-bars">${catBarsHtml}</div>
        </div>

        <div class="ba-panel">
          <div class="ba-panel-head">
            <h2 class="ba-panel-title">月度产出趋势</h2>
            <span class="ba-panel-tip">点击柱子定位当月</span>
          </div>
          <div class="ba-mbar-chart">${monthBarsHtml}</div>
        </div>
      </div>
    </section>

    <!-- Part 3: Paper Quick-Lookup Table -->
    <section class="ba-panel ba-table-section" id="ba-table-section" aria-label="论文速查表">
      <div class="ba-table-head">
        <div>
          <h2 class="ba-panel-title">论文速查表</h2>
          <p class="ba-panel-sub">按机构、专题、arXiv 编号或核心结论即时检索，一键直达精读笔记与原文</p>
        </div>
        <div class="ba-kind-tabs" role="tablist" aria-label="切换内容范围">
          <button type="button" class="ba-kind-btn is-active" data-kind-tab="paper">论文精读 <span>${paperCount}</span></button>
          <button type="button" class="ba-kind-btn" data-kind-tab="note">技术解析 <span>${noteCount}</span></button>
          <button type="button" class="ba-kind-btn" data-kind-tab="all">全部 <span>${totalCount}</span></button>
        </div>
      </div>

      <div class="ba-table-toolbar">
        <label class="ba-search-wrap" aria-label="搜索论文或笔记">
          <i class="iconfont icon-search" aria-hidden="true"></i>
          <input id="ba-search-input" type="search" placeholder="搜标题、机构（如 DeepMind / 清华）、arXiv、结论…" autocomplete="off" />
          <button type="button" id="ba-search-clear" class="ba-clear-btn" aria-label="清空" hidden>×</button>
        </label>

        <div class="ba-selects">
          <select id="ba-cat-select" class="ba-select" aria-label="按专题筛选">
            ${topicOptionsHtml}
          </select>
          <select id="ba-sort-select" class="ba-select" aria-label="排序方式">
            <option value="date-desc">发布日期：最新优先</option>
            <option value="date-asc">发布日期：最早优先</option>
            <option value="cat">按专题归类</option>
            <option value="title">按标题 A–Z</option>
          </select>
        </div>
      </div>

      ${orgPillsHtml}

      <div class="ba-active-filters" id="ba-active-filters">
        <span class="ba-result-count" id="ba-result-count">当前显示 <strong>${paperCount}</strong> 篇</span>
        <div class="ba-filter-tags" id="ba-filter-tags"></div>
        <button type="button" class="ba-reset-btn" id="ba-reset-btn" hidden>重置筛选</button>
      </div>

      <div class="ba-table-wrap">
        <table class="ba-table">
          <thead>
            <tr>
              <th class="ba-col-date">日期</th>
              <th class="ba-col-title">标题与核心结论</th>
              <th class="ba-col-org">机构 / 来源</th>
              <th class="ba-col-cat">专题方向</th>
              <th class="ba-col-source">原文</th>
            </tr>
          </thead>
          <tbody id="ba-table-body">
            ${rowsHtml}
          </tbody>
        </table>
      </div>

      <div class="ba-empty" id="ba-empty" hidden>
        <p>当前筛选条件下没有找到匹配文章。</p>
        <button type="button" class="ba-reset-btn" id="ba-empty-reset">清除筛选条件</button>
      </div>
    </section>

    <div class="ba-tooltip" id="ba-tooltip" role="tooltip" hidden></div>
  </div>`;
}

// Register after hexo-generator-archive so /archives/index.html uses the dashboard + paper table.
hexo.on('generateBefore', function () {
  hexo.extend.generator.register('archive', function (locals) {
    const theme = this.theme.config || {};
    return {
      path: `${this.config.archive_dir || 'archives'}/index.html`,
      layout: 'page',
      data: {
        title: '足迹 · 论文库',
        subtitle: '阅读足迹 & 论文速查库',
        banner_img: (theme.archive && theme.archive.banner_img) || '/img/bg.png',
        banner_img_height: (theme.archive && theme.archive.banner_img_height) || 30,
        banner_mask_alpha: (theme.archive && theme.archive.banner_mask_alpha) ?? 0.42,
        comments: false,
        content: buildArchiveDashboardHtml(locals, this),
      },
    };
  });
});
