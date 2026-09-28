const buttons = document.querySelectorAll(".tab-button");
const panels = document.querySelectorAll(".table-panel");

buttons.forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.table;

    buttons.forEach((item) => item.classList.remove("active"));
    panels.forEach((panel) => panel.classList.remove("active"));

    button.classList.add("active");
    document.getElementById(target)?.classList.add("active");
  });
});

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;

    await navigator.clipboard.writeText(target.textContent.trim());
    button.classList.add("copied");
    setTimeout(() => button.classList.remove("copied"), 1200);
  });
});

document.querySelectorAll(".play-pair-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const card = button.closest(".video-card");
    const videos = Array.from(card?.querySelectorAll("video") || []);

    if (button.classList.contains("is-playing")) {
      videos.forEach((video) => {
        video.pause();
        video.currentTime = 0;
      });
      button.classList.remove("is-playing");
      button.textContent = "Play Both";
      return;
    }

    videos.forEach((video) => {
      video.currentTime = 0;
      video.muted = true;
    });

    await Promise.allSettled(videos.map((video) => video.play()));
    button.classList.add("is-playing");
    button.textContent = "Stop Both";
  });
});
