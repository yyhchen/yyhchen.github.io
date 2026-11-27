'use strict';

const { escapeHTML, url_for: urlFor } = require('hexo-util');

const PAPER_SLUG_MAP = new Map([
  ['智能体与Harness', 'agents'],
  ['强化学习与后训练', 'reinforcement-learning'],
  ['模型架构与推理效率', 'architecture'],
  ['安全与可解释性', 'safety'],
  ['评测与验证', 'evaluation'],
  ['多模态与具身智能', 'multimodal'],
  ['理论与数学推理', 'theory'],
]);

function renderBreadcrumb(post, context) {
  const homeUrl = escapeHTML(urlFor.call(context, '/'));
  const topCat = post.categories && post.categories.toArray().find((entry) => !entry.parent);

  if (post.paper_reading) {
    const catName = topCat ? topCat.name : '论文精读';
    const slug = PAPER_SLUG_MAP.get(catName) || 'papers';
    const papersUrl = escapeHTML(urlFor.call(context, '/#cat=papers'));
    const catUrl = escapeHTML(urlFor.call(context, `/#cat=${slug}`));

    return `<nav class="post-breadcrumb" aria-label="文章位置">
      <a href="${homeUrl}">首页</a>
      <span class="bc-sep">/</span>
      <a href="${papersUrl}">论文精读</a>
      <span class="bc-sep">/</span>
      <a class="bc-current" href="${catUrl}">${escapeHTML(catName)}</a>
      <span class="bc-kind is-paper">论文精读</span>
    </nav>`;
  }

  const catName = topCat ? topCat.name : '技术解析';
  const catUrl = escapeHTML(urlFor.call(context, '/#cat=tech-notes'));
  return `<nav class="post-breadcrumb" aria-label="文章位置">
    <a href="${homeUrl}">首页</a>
    <span class="bc-sep">/</span>
    <a class="bc-current" href="${catUrl}">${escapeHTML(catName)}</a>
    <span class="bc-kind is-note">技术解析</span>
  </nav>`;
}

// Inject post breadcrumbs for every post, and apply Obsidian reading styles to paper notes.
hexo.extend.filter.register('after_post_render', function (post) {
  if (!post.content || post.layout !== 'post') return post;

  const breadcrumb = renderBreadcrumb(post, this);

  if (post.paper_reading) {
    const content = post.content.replace(
      /<table(?:\s[^>]*)?>[\s\S]*?<\/table>/gi,
      (table) => `<div class="paper-table-scroll" role="region" aria-label="论文表格" tabindex="0">${table}</div>`
    );
    post.content = `<div class="paper-reading">\n${breadcrumb}\n${content}\n</div>`;
  } else {
    post.content = `${breadcrumb}\n${post.content}`;
  }

  return post;
}, 40);
