"use strict";

const state = {
  manifest: null,
  metrics: null,
  videos: [],
  videoId: null,
  skipEnabled: false,
  skippedIntervals: new Set(),
  timelineDragging: false,
  rafId: null,
  pulseTimer: null,
  lastTime: 0,
};

const el = {};
const IDS = [
  "videoSelect",
  "videoShell",
  "videoPlayer",
  "adSkipButton",
  "playPulse",
  "playbackWarning",
  "currentTimeText",
  "durationText",
  "adTimeline",
  "timelineLayers",
  "timeMarker",
  "overallMetricsList",
  "selectedVideoMetricTitle",
  "videoMetricsList",
];

const METRIC_ROWS = [
  {label: "광고 구간 포착률", key: "ad_capture_rate_pct", format: formatPercent, title: "실제 광고와 예측 광고가 겹친 시간 / 실제 광고 전체 시간"},
  {label: "예측 광고 정밀도", key: "prediction_precision_pct", format: formatPercent, title: "실제 광고와 예측 광고가 겹친 시간 / 예측 광고 전체 시간"},
  {label: "평균 시작 오차", key: "mean_start_error_sec", format: formatSeconds, title: "예측 시작 시점과 실제 시작 시점의 평균 절대 차이"},
  {label: "평균 종료 오차", key: "mean_end_error_sec", format: formatSeconds, title: "예측 종료 시점과 실제 종료 시점의 평균 절대 차이"},
  {label: "비광고 오탐 시간", key: "false_positive_duration_sec", format: formatDurationWithSeconds, title: "전체 성능은 영상별 평균, 선택 영상은 해당 영상의 오탐 시간"},
];

document.addEventListener("DOMContentLoaded", init);

async function init() {
  cacheElements();
  bindEvents();
  state.manifest = await loadManifest();
  state.metrics = await loadMetrics();
  state.videos = state.manifest.videos || [];
  state.videoId = Number(state.manifest.default_video_id || (state.videos[0] && state.videos[0].video_id));
  renderVideoSelector();
  loadSelectedVideo();
}

function cacheElements() {
  IDS.forEach(id => {
    el[id] = document.getElementById(id);
  });
}

async function loadManifest() {
  if (window.DEMO_VIEWER_MANIFEST) {
    return window.DEMO_VIEWER_MANIFEST;
  }
  return fetch("demo_viewer_manifest.json", {cache: "no-store"}).then(response => response.json());
}

async function loadMetrics() {
  if (window.DEMO_VIEWER_METRICS) {
    return window.DEMO_VIEWER_METRICS;
  }
  try {
    return await fetch("demo_viewer_metrics.json", {cache: "no-store"}).then(response => response.json());
  } catch (error) {
    return null;
  }
}

function bindEvents() {
  el.videoSelect.addEventListener("change", () => {
    state.videoId = Number(el.videoSelect.value);
    loadSelectedVideo();
  });

  el.videoShell.addEventListener("click", handleVideoShellClick);

  el.adSkipButton.addEventListener("click", event => {
    event.stopPropagation();
    setSkipEnabled(!state.skipEnabled);
  });

  el.videoPlayer.addEventListener("loadedmetadata", () => {
    updatePlaybackUi();
    renderTimeline();
  });
  el.videoPlayer.addEventListener("timeupdate", handleTimeUpdate);
  el.videoPlayer.addEventListener("play", startMarkerLoop);
  el.videoPlayer.addEventListener("pause", updatePlaybackUi);
  el.videoPlayer.addEventListener("ended", updatePlaybackUi);

  el.adTimeline.addEventListener("pointerdown", beginTimelineSeek);
  document.addEventListener("keydown", handleKeyboardShortcuts);
  window.addEventListener("resize", renderTimeline);
}

function renderVideoSelector() {
  el.videoSelect.innerHTML = "";
  state.videos.forEach(video => {
    const option = document.createElement("option");
    option.value = String(video.video_id);
    option.textContent = "video " + video.video_id;
    el.videoSelect.appendChild(option);
  });
  el.videoSelect.value = String(state.videoId);
}

function loadSelectedVideo() {
  const video = currentVideo();
  if (!video) return;
  el.videoPlayer.pause();
  el.videoPlayer.controls = false;
  state.skippedIntervals.clear();
  state.lastTime = 0;
  setSkipEnabled(false, {skipImmediateCheck: true});

  const nextSrc = videoSourceFor(video);
  if (el.videoPlayer.getAttribute("src") !== nextSrc) {
    el.videoPlayer.src = nextSrc;
    el.videoPlayer.load();
  }

  const playable = Boolean(video.playable && nextSrc);
  el.playbackWarning.hidden = playable;
  el.playbackWarning.textContent = playable ? "" : "등록된 영상 경로를 재생할 수 없습니다.";
  el.currentTimeText.textContent = fmt(0);
  el.durationText.textContent = fmt(duration());
  renderTimeline();
  renderMetricsPanels();
  updatePlaybackUi();
}

function currentVideo() {
  return state.videos.find(video => Number(video.video_id) === Number(state.videoId)) || state.videos[0] || null;
}

function videoSourceFor(video) {
  if (!video) return "";
  if (!video.playable) return video.video_url || "";
  if (window.location.protocol === "http:" || window.location.protocol === "https:") {
    return "/media/" + Number(video.video_id);
  }
  return video.video_url || video.video_path || "";
}

function handleVideoShellClick(event) {
  if (event.target.closest("button")) return;
  const rect = el.videoShell.getBoundingClientRect();
  if (rect.width <= 0) return;
  const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  if (ratio < 0.25) {
    seek((el.videoPlayer.currentTime || 0) - 5);
    showPulse("rewind");
    return;
  }
  if (ratio > 0.75) {
    seek((el.videoPlayer.currentTime || 0) + 5);
    showPulse("forward");
    return;
  }
  togglePlay();
}

function handleKeyboardShortcuts(event) {
  if (shouldIgnoreKeyboardTarget(event.target)) return;
  if (event.key === "ArrowRight") {
    event.preventDefault();
    seek((el.videoPlayer.currentTime || 0) - 5);
    showPulse("rewind");
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    seek((el.videoPlayer.currentTime || 0) + 5);
    showPulse("forward");
  } else if (event.code === "Space" || event.key === " " || event.key === "Spacebar") {
    event.preventDefault();
    togglePlay();
  }
}

function shouldIgnoreKeyboardTarget(target) {
  const tagName = String((target && target.tagName) || "").toLowerCase();
  return ["button", "select", "input", "textarea"].includes(tagName) || Boolean(target && target.isContentEditable);
}

function togglePlay() {
  const player = el.videoPlayer;
  if (player.paused || player.ended) {
    player.play().then(() => showPulse("playing")).catch(() => {});
  } else {
    player.pause();
    showPulse("paused");
  }
}

function showPulse(mode) {
  window.clearTimeout(state.pulseTimer);
  el.playPulse.hidden = true;
  el.playPulse.className = "play-pulse is-" + mode;
  void el.playPulse.offsetWidth;
  el.playPulse.hidden = false;
  state.pulseTimer = window.setTimeout(() => {
    el.playPulse.hidden = true;
  }, 560);
}

function setSkipEnabled(enabled, options = {}) {
  state.skipEnabled = Boolean(enabled);
  el.adSkipButton.classList.toggle("is-on", state.skipEnabled);
  el.adSkipButton.setAttribute("aria-pressed", String(state.skipEnabled));
  el.adSkipButton.textContent = state.skipEnabled ? "광고 스킵 ON" : "광고 스킵";
  if (!state.skipEnabled) {
    state.skippedIntervals.clear();
  } else if (!options.skipImmediateCheck) {
    maybeSkipAd();
  }
}

function handleTimeUpdate() {
  const current = Number(el.videoPlayer.currentTime) || 0;
  resetSkipGuards(current);
  maybeSkipAd();
  state.lastTime = Number(el.videoPlayer.currentTime) || current;
  updatePlaybackUi();
}

function maybeSkipAd() {
  if (!state.skipEnabled) return false;
  const player = el.videoPlayer;
  const current = Number(player.currentTime) || 0;
  const intervals = skipIntervalsForCurrentVideo();
  for (let index = 0; index < intervals.length; index += 1) {
    const interval = intervals[index];
    const key = intervalKey(interval, index);
    const start = Number(interval.start_sec);
    const end = Number(interval.end_sec);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
    if (current >= start && current < end - 0.05 && !state.skippedIntervals.has(key)) {
      state.skippedIntervals.add(key);
      const wasPlaying = !player.paused && !player.ended;
      const target = Math.min(end + 0.2, timelineDuration());
      player.currentTime = target;
      if (wasPlaying) {
        player.play().catch(() => {});
      }
      updatePlaybackUi();
      return true;
    }
  }
  return false;
}

function resetSkipGuards(current) {
  const intervals = skipIntervalsForCurrentVideo();
  intervals.forEach((interval, index) => {
    if (current < Number(interval.start_sec) - 0.2) {
      state.skippedIntervals.delete(intervalKey(interval, index));
    }
  });
}

function skipIntervalsForCurrentVideo() {
  const video = currentVideo();
  return mergeIntervals((video && video.predicted_intervals) || []);
}

function mergeIntervals(intervals) {
  const sorted = (intervals || [])
    .map(item => ({
      start_sec: Number(item.start_sec),
      end_sec: Number(item.end_sec),
    }))
    .filter(item => Number.isFinite(item.start_sec) && Number.isFinite(item.end_sec) && item.end_sec > item.start_sec)
    .sort((a, b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec);
  const merged = [];
  sorted.forEach(item => {
    const last = merged[merged.length - 1];
    if (!last || item.start_sec > last.end_sec + 0.5) {
      merged.push({...item});
    } else {
      last.end_sec = Math.max(last.end_sec, item.end_sec);
    }
  });
  return merged;
}

function intervalKey(interval, index) {
  return String(index) + ":" + Number(interval.start_sec).toFixed(3) + ":" + Number(interval.end_sec).toFixed(3);
}

function renderTimeline() {
  const video = currentVideo();
  if (!video) return;
  el.timelineLayers.innerHTML = "";
  const dur = duration();
  if (dur <= 0) {
    updatePlaybackUi();
    return;
  }
  addSegmentLayer("예측 광고 구간", video.predicted_intervals || [], "segment-predicted", dur);
  addSegmentLayer("실제 광고 구간", video.actual_intervals || [], "segment-actual", dur);
  updatePlaybackUi();
}

function addSegmentLayer(label, items, className, dur) {
  const layer = document.createElement("div");
  layer.className = "timeline-layer";
  (items || []).forEach(item => {
    const start = Number(item.start_sec);
    const end = Number(item.end_sec);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
    const segment = document.createElement("div");
    segment.className = "timeline-segment " + className;
    segment.style.left = String(clamp(start / dur, 0, 1) * 100) + "%";
    segment.style.width = String(Math.max(0.15, ((Math.min(end, dur) - Math.max(start, 0)) / dur) * 100)) + "%";
    segment.title = label + " " + fmt(start) + "-" + fmt(end);
    layer.appendChild(segment);
  });
  el.timelineLayers.appendChild(layer);
}

function beginTimelineSeek(event) {
  event.preventDefault();
  state.timelineDragging = true;
  if (el.adTimeline.setPointerCapture) {
    el.adTimeline.setPointerCapture(event.pointerId);
  }
  seekFromTimelinePointer(event);

  const move = moveEvent => {
    if (state.timelineDragging) {
      seekFromTimelinePointer(moveEvent);
    }
  };
  const stop = stopEvent => {
    state.timelineDragging = false;
    if (el.adTimeline.releasePointerCapture) {
      try {
        el.adTimeline.releasePointerCapture(stopEvent.pointerId);
      } catch (error) {
        // 브라우저가 pointer capture를 먼저 해제한 경우를 허용한다.
      }
    }
    el.adTimeline.removeEventListener("pointermove", move);
    el.adTimeline.removeEventListener("pointerup", stop);
    el.adTimeline.removeEventListener("pointercancel", stop);
  };
  el.adTimeline.addEventListener("pointermove", move);
  el.adTimeline.addEventListener("pointerup", stop);
  el.adTimeline.addEventListener("pointercancel", stop);
}

function seekFromTimelinePointer(event) {
  const dur = timelineDuration();
  const rect = el.adTimeline.getBoundingClientRect();
  if (dur <= 0 || rect.width <= 0) return;
  const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  seek(ratio * dur);
}

function seek(seconds) {
  const dur = timelineDuration();
  const nextTime = clamp(Number(seconds) || 0, 0, dur || duration());
  el.videoPlayer.currentTime = nextTime;
  state.lastTime = nextTime;
  resetSkipGuards(nextTime);
  updatePlaybackUi();
}

function renderMetricsPanels() {
  const metrics = state.metrics || {};
  renderMetricList(el.overallMetricsList, metrics.overall_metrics || null);
  if (el.selectedVideoMetricTitle) {
    el.selectedVideoMetricTitle.textContent = "video " + state.videoId;
  }
  const byVideo = metrics.metrics_by_video || {};
  renderMetricList(el.videoMetricsList, byVideo[String(state.videoId)] || null);
}

function renderMetricList(container, metrics) {
  if (!container) return;
  container.innerHTML = "";
  if (!metrics) {
    const empty = document.createElement("div");
    empty.className = "metric-empty";
    empty.textContent = "계산 불가";
    container.appendChild(empty);
    return;
  }
  METRIC_ROWS.forEach(row => {
    const item = document.createElement("div");
    item.className = "metric-item";
    item.title = row.title;
    const label = document.createElement("span");
    label.className = "metric-label";
    label.textContent = row.label;
    const value = document.createElement("strong");
    value.className = "metric-value";
    value.textContent = row.format(metrics[row.key]);
    item.appendChild(label);
    item.appendChild(value);
    container.appendChild(item);
  });
}

function updatePlaybackUi() {
  const current = Number(el.videoPlayer.currentTime) || 0;
  const dur = timelineDuration();
  el.currentTimeText.textContent = fmt(current);
  el.durationText.textContent = fmt(dur);
  setTimelineAria(current, dur);
  updateMarker(current, dur);
}

function updateMarker(current, dur) {
  const usableDuration = dur || timelineDuration();
  const percent = usableDuration > 0 ? (clamp(current || 0, 0, usableDuration) / usableDuration) * 100 : 0;
  el.timeMarker.style.left = String(percent) + "%";
}

function setTimelineAria(current, dur) {
  el.adTimeline.setAttribute("aria-valuemin", "0");
  el.adTimeline.setAttribute("aria-valuemax", String(Math.max(0, dur || 0)));
  el.adTimeline.setAttribute("aria-valuenow", String(clamp(current || 0, 0, dur || 0)));
  el.adTimeline.setAttribute("aria-valuetext", fmt(current) + " / " + fmt(dur));
}

function startMarkerLoop() {
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
  }
  const tick = () => {
    updatePlaybackUi();
    if (!el.videoPlayer.paused && !el.videoPlayer.ended) {
      state.rafId = requestAnimationFrame(tick);
    }
  };
  state.rafId = requestAnimationFrame(tick);
}

function duration() {
  const video = currentVideo();
  return Number((video && (video.video_duration_sec || video.duration_sec)) || 1);
}

function timelineDuration() {
  const mediaDuration = Number.isFinite(el.videoPlayer.duration) ? Number(el.videoPlayer.duration) : 0;
  return mediaDuration > 0 ? mediaDuration : duration();
}

function fmt(seconds) {
  const sec = Math.max(0, Math.round(Number(seconds) || 0));
  return String(Math.floor(sec / 60)).padStart(2, "0") + ":" + String(sec % 60).padStart(2, "0");
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) + "%" : "계산 불가";
}

function formatSeconds(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) + "초" : "계산 불가";
}

function formatDurationWithSeconds(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "계산 불가";
  const rounded = Math.max(0, Math.round(number));
  const minutes = Math.floor(rounded / 60);
  const seconds = String(rounded % 60).padStart(2, "0");
  return String(minutes) + ":" + seconds + " (" + number.toFixed(1) + "초)";
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
