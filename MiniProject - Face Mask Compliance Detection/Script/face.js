/**
 * face.js — Live dashboard logic
 * Polls /stats every second, updates counters + glow animations.
 */

(() => {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────
  const statusDot    = document.getElementById('statusDot');
  const statusText   = document.getElementById('statusText');
  const statusPill   = document.getElementById('connectionStatus');
  const uptimeBadge  = document.getElementById('uptimeBadge');
  const fpsBadge     = document.getElementById('fpsBadge');
  const videoWrapper = document.getElementById('videoWrapper');

  const statTotal     = document.getElementById('statTotal');
  const statCorrect   = document.getElementById('statCorrect');
  const statIncorrect = document.getElementById('statIncorrect');
  const statNoMask    = document.getElementById('statNoMask');

  const barCorrect   = document.getElementById('barCorrect');
  const barIncorrect = document.getElementById('barIncorrect');
  const barNoMask    = document.getElementById('barNoMask');

  // ── State ─────────────────────────────────────────────────
  let isOnline = false;
  let startTime = Date.now();

  // ── Uptime Timer ──────────────────────────────────────────
  function formatTime(seconds) {
    const h = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
    const s = String(seconds % 60).padStart(2, '0');
    return `⏱ ${h}:${m}:${s}`;
  }

  setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    uptimeBadge.textContent = formatTime(elapsed);
  }, 1000);

  // ── Animated counter ─────────────────────────────────────
  function animateValue(el, newVal) {
    const current = parseInt(el.textContent) || 0;
    if (current === newVal) return;
    const step = newVal > current ? 1 : -1;
    const diff = Math.abs(newVal - current);
    const delay = diff > 5 ? 20 : 60;
    let val = current;
    const interval = setInterval(() => {
      val += step;
      el.textContent = val;
      if (val === newVal) clearInterval(interval);
    }, delay);
  }

  // ── Glow state for video ──────────────────────────────────
  function setVideoGlow(dominant) {
    videoWrapper.classList.remove('glow-green', 'glow-orange', 'glow-red');
    if (dominant === 'proper_mask')   videoWrapper.classList.add('glow-green');
    else if (dominant === 'incorrect_mask') videoWrapper.classList.add('glow-orange');
    else if (dominant === 'no_mask') videoWrapper.classList.add('glow-red');
  }

  // ── Update UI from stats payload ──────────────────────────
  function applyStats(data) {
    const total    = data.total_faces   || 0;
    const correct  = data.proper_mask   || 0;
    const wrong    = data.incorrect_mask|| 0;
    const none     = data.no_mask       || 0;

    animateValue(statTotal,     total);
    animateValue(statCorrect,   correct);
    animateValue(statIncorrect, wrong);
    animateValue(statNoMask,    none);

    // Progress bars (% of total faces)
    if (total > 0) {
      barCorrect.style.width   = `${(correct / total) * 100}%`;
      barIncorrect.style.width = `${(wrong   / total) * 100}%`;
      barNoMask.style.width    = `${(none    / total) * 100}%`;
    } else {
      barCorrect.style.width = barIncorrect.style.width = barNoMask.style.width = '0%';
    }

    // FPS badge
    fpsBadge.textContent = data.fps ? `${data.fps} FPS` : '-- FPS';

    // Dominant label → glow
    if (total === 0) {
      videoWrapper.classList.remove('glow-green', 'glow-orange', 'glow-red');
    } else {
      const counts = { proper_mask: correct, incorrect_mask: wrong, no_mask: none };
      const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      setVideoGlow(dominant);
    }
  }

  // ── Connection status helpers ─────────────────────────────
  function setOnline() {
    if (isOnline) return;
    isOnline = true;
    statusDot.className = 'status-dot online';
    statusText.textContent = 'Live';
    statusPill.className = 'status-pill online';
  }

  function setOffline() {
    if (!isOnline) return;
    isOnline = false;
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Disconnected';
    statusPill.className = 'status-pill offline';
    videoWrapper.classList.remove('glow-green', 'glow-orange', 'glow-red');
  }

  // ── Polling loop ──────────────────────────────────────────
  async function pollStats() {
    try {
      const res = await fetch('/stats', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setOnline();
      applyStats(data);
    } catch (err) {
      setOffline();
      console.warn('[Stats] Poll failed:', err.message);
    }
  }

  // ── Video error handler ──────────────────────────────────
  window.handleVideoError = function () {
    setOffline();
    console.warn('[Video] Stream error — camera may be unavailable.');
  };

  // ── Init ─────────────────────────────────────────────────
  pollStats();
  setInterval(pollStats, 1000);
})();
