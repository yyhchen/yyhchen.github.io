// 主题切换功能
document.addEventListener('DOMContentLoaded', function() {
  const themeSwitch = document.querySelector('.theme-switch');
  const htmlElement = document.documentElement;
  
  // 检查本地存储中的主题设置
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    htmlElement.setAttribute('data-theme', savedTheme);
  } else {
    // 检查系统偏好
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const defaultTheme = prefersDarkMode ? 'dark' : 'light';
    htmlElement.setAttribute('data-theme', defaultTheme);
    localStorage.setItem('theme', defaultTheme);
  }
  
  // 添加主题切换事件
  if (themeSwitch) {
    themeSwitch.addEventListener('click', function() {
      const currentTheme = htmlElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      // 添加过渡类
      document.body.classList.add('theme-transition');
      
      // 设置新主题
      htmlElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      
      // 移除过渡类
      setTimeout(() => {
        document.body.classList.remove('theme-transition');
      }, 300);
    });
  }
});

// 添加平滑滚动效果
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: 'smooth'
        });
      }
    });
  });
});

// 响应式导航菜单
document.addEventListener('DOMContentLoaded', function() {
  const header = document.querySelector('.header');
  const navList = document.querySelector('.nav-list');
  
  // 创建移动端菜单按钮
  const mobileMenuBtn = document.createElement('div');
  mobileMenuBtn.className = 'mobile-menu-btn';
  mobileMenuBtn.innerHTML = '<span></span><span></span><span></span>';
  
  // 将按钮添加到头部
  if (header && window.innerWidth <= 768) {
    header.insertBefore(mobileMenuBtn, navList);
    
    // 添加点击事件
    mobileMenuBtn.addEventListener('click', function() {
      navList.classList.toggle('active');
      mobileMenuBtn.classList.toggle('active');
    });
  }
  
  // 窗口大小变化时处理
  window.addEventListener('resize', function() {
    if (window.innerWidth <= 768) {
      if (!header.contains(mobileMenuBtn)) {
        header.insertBefore(mobileMenuBtn, navList);
      }
    } else {
      if (header.contains(mobileMenuBtn)) {
        header.removeChild(mobileMenuBtn);
      }
      navList.classList.remove('active');
    }
  });
});