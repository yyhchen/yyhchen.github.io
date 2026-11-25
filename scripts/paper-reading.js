// Apply the Obsidian reading styles only to the imported paper notes.
hexo.extend.filter.register('after_post_render', (post) => {
  if (!post.paper_reading || !post.content) return post;

  const content = post.content.replace(/<table(?:\s[^>]*)?>[\s\S]*?<\/table>/gi,
    (table) => `<div class="paper-table-scroll" role="region" aria-label="论文表格" tabindex="0">${table}</div>`);
  post.content = `<div class="paper-reading">\n${content}\n</div>`;
  return post;
}, 40);
