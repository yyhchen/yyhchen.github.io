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

function renderNavigation(groups) {
  const links = groups.map(({ id, name, posts }) =>
    `<li><a href="#${id}"><span>${escapeHTML(name)}</span><span class="paper-index-count">${posts.length}</span></a></li>`
  ).join('\n');

  return `<nav class="paper-index-nav" aria-labelledby="paper-categories-title">
    <h2 id="paper-categories-title">按分类跳转</h2>
    <ul>${links}</ul>
  </nav>`;
}

function renderSections(groups, titleCounts, context) {
  return groups.map(({ id, name, posts }) => {
    const rows = posts.map((post) => {
      const url = escapeHTML(urlFor.call(context, post.path));
      const title = escapeHTML(post.title.replace(/^Paper精读[｜|]\s*/u, ''));
      const date = post.date.format('YYYY-MM-DD');
      const time = titleCounts.get(post.title) > 1 ? `<span>${post.date.format('HH:mm')}</span>` : '';

      return `<li><a href="${url}"><time datetime="${post.date.format()}">${date}${time}</time><span>${title}</span></a></li>`;
    }).join('\n');

    return `<section id="${id}" class="paper-index-section" aria-labelledby="${id}-title" tabindex="-1">
      <div class="paper-index-heading">
        <h2 id="${id}-title">${escapeHTML(name)} <span class="paper-index-count">${posts.length} 篇</span></h2>
        <a href="#paper-categories">返回分类</a>
      </div>
      <ul class="paper-index-posts">${rows}</ul>
    </section>`;
  }).join('\n');
}

hexo.extend.generator.register('paper-reading-index', function (locals) {
  const groups = new Map([...categoryOrder.keys()].map((name) => [name, []]));
  const titleCounts = new Map();
  const posts = locals.posts.find({ paper_reading: true }).sort('-date').toArray();

  for (const post of posts) {
    const category = post.categories.toArray().find((entry) => !entry.parent);
    const name = category ? category.name : '未分类';
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(post);
    titleCounts.set(post.title, (titleCounts.get(post.title) || 0) + 1);
  }

  const categories = [...groups.entries()]
    .filter(([, entries]) => entries.length)
    .map(([name, entries], index) => ({
      id: 'papers-' + (categoryOrder.get(name) || `category-${index + 1}`),
      name,
      posts: entries,
    }));

  const content = `<div id="paper-categories" class="paper-index" tabindex="-1">
    ${renderNavigation(categories)}
    <div class="paper-index-list">${renderSections(categories, titleCounts, this)}</div>
  </div>`;

  return {
    path: 'paper-reading/index.html',
    layout: 'page',
    data: {
      title: '论文精读',
      subtitle: '论文精读',
      banner_img_height: 31,
      comments: false,
      content,
    },
  };
});
