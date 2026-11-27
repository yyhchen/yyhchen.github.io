'use strict';

const { escapeHTML, url_for: urlFor } = require('hexo-util');

const categoryOrder = new Map([
  ['智能体与Harness', 'agents'],
  ['强化学习与后训练', 'reinforcement-learning'],
  ['模型架构与推理效率', 'architecture'],
  ['安全与可解释性', 'safety'],
  ['评测与验证', 'evaluation'],
  ['多模态与具身智能', 'multimodal'],
  ['理论与数学推理', 'theory'],
]);

function renderNavigation(noteGroup, paperGroups) {
  const noteLink = noteGroup
    ? `<div class="paper-index-group-label">基础专栏</div>
       <ul>
         <li><a href="#${noteGroup.id}"><span>${escapeHTML(noteGroup.name)}</span><span class="paper-index-count">${noteGroup.posts.length}</span></a></li>
       </ul>`
    : '';

  const paperLinks = paperGroups
    .map(
      ({ id, name, posts }) =>
        `<li><a href="#${id}"><span>${escapeHTML(name)}</span><span class="paper-index-count">${posts.length}</span></a></li>`
    )
    .join('\n');

  return `<nav class="paper-index-nav" aria-labelledby="paper-categories-title">
    <h2 id="paper-categories-title">按专题跳转</h2>
    ${noteLink}
    <div class="paper-index-group-label">论文精读 · 主题方向</div>
    <ul>${paperLinks}</ul>
  </nav>`;
}

function renderSections(groups, titleCounts, context) {
  return groups
    .map(({ id, name, posts, kind }) => {
      const badge = kind === 'note' ? '<span class="paper-index-badge is-note">基础学习</span>' : '';
      const rows = posts
        .map((post) => {
          const url = escapeHTML(urlFor.call(context, post.path));
          const title = escapeHTML(post.title.replace(/^Paper精读[｜|]\s*/u, ''));
          const date = post.date.format('YYYY-MM-DD');
          const time = titleCounts.get(post.title) > 1 ? `<span>${post.date.format('HH:mm')}</span>` : '';

          return `<li><a href="${url}"><time datetime="${post.date.format()}">${date}${time}</time><span>${title}</span></a></li>`;
        })
        .join('\n');

      return `<section id="${id}" class="paper-index-section" aria-labelledby="${id}-title" tabindex="-1">
      <div class="paper-index-heading">
        <h2 id="${id}-title">${escapeHTML(name)}${badge} <span class="paper-index-count">${posts.length} 篇</span></h2>
        <a href="#paper-categories">返回目录</a>
      </div>
      <ul class="paper-index-posts">${rows}</ul>
    </section>`;
    })
    .join('\n');
}

hexo.extend.generator.register('paper-reading-index', function (locals) {
  const allPosts = locals.posts.filter((p) => !p.hide).sort('-date').toArray();
  const titleCounts = new Map();
  for (const post of allPosts) {
    titleCounts.set(post.title, (titleCounts.get(post.title) || 0) + 1);
  }

  const notePosts = allPosts.filter((p) => !p.paper_reading);
  const paperPosts = allPosts.filter((p) => p.paper_reading);

  const paperMap = new Map([...categoryOrder.keys()].map((name) => [name, []]));
  for (const post of paperPosts) {
    const category = post.categories.toArray().find((entry) => !entry.parent);
    const name = category ? category.name : '其他论文';
    if (!paperMap.has(name)) paperMap.set(name, []);
    paperMap.get(name).push(post);
  }

  const noteGroup = notePosts.length
    ? { id: 'section-tech-notes', name: '技术解析', posts: notePosts, kind: 'note' }
    : null;

  const paperGroups = [...paperMap.entries()]
    .filter(([, entries]) => entries.length)
    .map(([name, entries], index) => ({
      id: 'papers-' + (categoryOrder.get(name) || `category-${index + 1}`),
      name,
      posts: entries,
      kind: 'paper',
    }));

  const allGroups = noteGroup ? [noteGroup, ...paperGroups] : paperGroups;

  const content = `<div id="paper-categories" class="paper-index" tabindex="-1">
    ${renderNavigation(noteGroup, paperGroups)}
    <div class="paper-index-list">${renderSections(allGroups, titleCounts, this)}</div>
  </div>`;

  return {
    path: 'paper-reading/index.html',
    layout: 'page',
    data: {
      title: '专题目录',
      subtitle: '专题目录',
      banner_img_height: 30,
      comments: false,
      content,
    },
  };
});
