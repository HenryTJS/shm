/* ============================================================
   跨机构对齐总览 — 数据来自 data/alignment.js
   ============================================================ */
(function () {
  'use strict';

  var D = window.ALIGN_DATA;
  if (!D) return;

  var COLORS = { C1: '#4c8dff', C2: '#25d4f0', C3: '#ff8a1f', C4: '#ff5c69' };
  var CFG = {
    ae: {
      key: 'ae', field: 't85', unit: '%',
      title: 'AE t85 · 29 specimens',
      foot: '每个点是一个试件；横线是批内中位。AE：C1 5 / C2 9 / C3 5 / C4 10，共 29 组。',
      batches: ['C1', 'C2', 'C3', 'C4'],
      labs: { C1: 'Lab A', C2: 'Lab A', C3: 'Lab B', C4: 'Lab B' },
      ymin: 60, ymax: 102, ticks: [60, 70, 80, 90, 100],
      fmt: function (v) { return v.toFixed(1) + '%'; }
    },
    gw: {
      key: 'gw', field: 'slope', unit: '',
      title: 'GW slope · 17 specimens',
      foot: '每个点是一个试件；横线是批内中位。GW：C1 5 / C3 5 / C4 7，共 17 组。',
      batches: ['C1', 'C3', 'C4'],
      labs: { C1: 'Lab A', C3: 'Lab B', C4: 'Lab B' },
      ymin: 0, ymax: 0.055, ticks: [0, 0.02, 0.04],
      fmt: function (v) { return v.toFixed(4); }
    }
  };

  var state = { metric: 'ae', points: [] };
  var cv = document.getElementById('cvAlign');
  var ctx = cv.getContext('2d');
  var tip = document.getElementById('chartTip');
  var wrap = cv.parentNode;

  function median(a) {
    var b = a.slice().sort(function (x, y) { return x - y; });
    var m = b.length >> 1;
    return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
  }

  function rows() { return state.metric === 'ae' ? D.ae : D.gw; }

  function draw() {
    var cfg = CFG[state.metric];
    var rect = wrap.getBoundingClientRect();
    var w = Math.max(320, rect.width - 4);
    var h = Math.max(240, rect.height - 4);
    var dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.width = w + 'px'; cv.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var L = 48, R = 18, T = 28, B = 42;
    var W = w - L - R, H = h - T - B;
    var data = rows();
    var batches = cfg.batches;

    function y(v) {
      return T + H - (v - cfg.ymin) / (cfg.ymax - cfg.ymin) * H;
    }
    function x(i) {
      return L + (i + 0.5) * (W / batches.length);
    }

    // grid + y labels
    ctx.font = '11px "Cascadia Mono", Consolas, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    cfg.ticks.forEach(function (t) {
      var yy = y(t);
      ctx.strokeStyle = 'rgba(37,212,240,.10)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(w - R, yy); ctx.stroke();
      ctx.fillStyle = '#5b7288';
      ctx.fillText(cfg.fmt(t), L - 8, yy);
    });

    // axes
    ctx.strokeStyle = 'rgba(198,214,226,.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(L, T); ctx.lineTo(L, T + H); ctx.lineTo(w - R, T + H);
    ctx.stroke();

    // x labels
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    batches.forEach(function (b, i) {
      var xx = x(i);
      ctx.fillStyle = '#c6d6e2';
      ctx.font = '12px "Cascadia Mono", Consolas, monospace';
      ctx.fillText(b, xx, T + H + 7);
      ctx.fillStyle = '#5b7288';
      ctx.font = '9.5px "Segoe UI", "Microsoft YaHei", sans-serif';
      ctx.fillText(cfg.labs[b], xx, T + H + 23);
    });

    // Lab divider (AE only)
    if (state.metric === 'ae') {
      var dx = L + 2 * (W / batches.length);
      ctx.strokeStyle = 'rgba(198,214,226,.28)';
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(dx, T); ctx.lineTo(dx, T + H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#8ba3b7';
      ctx.font = '10px "Segoe UI", "Microsoft YaHei", sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText('Lab A', L + W / 4, T - 8);
      ctx.fillText('Lab B', L + W * 0.75, T - 8);
    }

    // points + medians
    state.points = [];
    batches.forEach(function (b, bi) {
      var vals = data.filter(function (r) { return r.b === b; });
      var cx = x(bi);
      var span = Math.min(0.28, 0.55 / Math.max(1, vals.length));
      vals.forEach(function (r, j) {
        var off = (j - (vals.length - 1) / 2) * span * 2;
        var px = cx + off * (W / batches.length);
        var py = y(r[cfg.field]);
        state.points.push({ x: px, y: py, r: r, b: b, val: r[cfg.field] });
        ctx.beginPath();
        ctx.arc(px, py, 4.2, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[b] || '#25d4f0';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,.72)';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      });

      var m = median(vals.map(function (r) { return r[cfg.field]; }));
      var my = y(m);
      ctx.strokeStyle = '#eaf6ff';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(cx - 0.32 * (W / batches.length), my);
      ctx.lineTo(cx + 0.32 * (W / batches.length), my);
      ctx.stroke();

      ctx.fillStyle = '#eaf6ff';
      ctx.font = '10px "Cascadia Mono", Consolas, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText(cfg.fmt(m), cx, my - 5);
    });

    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = '#5b7288';
    ctx.font = '9.5px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.fillText(state.metric === 'ae' ? 'lifetime fraction at t85 (%)' : 'TOF drift slope (us / 100% life)', L, T - 16);
  }

  function showTip(ev) {
    var r = cv.getBoundingClientRect();
    var mx = ev.clientX - r.left, my = ev.clientY - r.top;
    var best = null, bestD = 14;
    state.points.forEach(function (p) {
      var d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestD) { bestD = d; best = p; }
    });
    if (!best) { tip.style.display = 'none'; return; }
    var cfg = CFG[state.metric];
    tip.innerHTML = '<b>' + best.r.g + '</b> · ' + best.b + ' · ' + cfg.fmt(best.val);
    tip.style.display = 'block';
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var left = Math.min(Math.max(4, best.x + 10), r.width - tw - 4);
    var top = Math.max(4, best.y - th - 8);
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }

  cv.addEventListener('mousemove', showTip);
  cv.addEventListener('mouseleave', function () { tip.style.display = 'none'; });

  // ---------------- η² ----------------
  var ETA = [
    { key: 'ae_t50', label: 'AE t50', short: '0.000', cls: 'ae' },
    { key: 'ae_t85', label: 'AE t85', short: '0.057', cls: 'ae' },
    { key: 'gw', label: 'GW slope', short: '0.000', cls: 'gw' }
  ];
  var maxEta = 0.15;

  function renderEta() {
    var host = document.getElementById('etaRows');
    host.innerHTML = ETA.map(function (e) {
      var g = D.eta[e.key];
      var c = g.campaign, u = g.university;
      var cw = Math.min(100, c.eta / maxEta * 100);
      var uw = Math.min(100, u.eta / maxEta * 100);
      var cn = Math.min(100, c.null / maxEta * 100);
      var un = Math.min(100, u.null / maxEta * 100);
      return '<div class="eta-row">' +
        '<div class="eta-label">' + e.label + '</div>' +
        '<div class="eta-track">' +
          '<i class="camp" style="width:' + cw.toFixed(2) + '%"></i>' +
          '<i class="univ" style="width:' + uw.toFixed(2) + '%"></i>' +
          '<i class="null" style="left:' + cn.toFixed(2) + '%"></i>' +
          '<i class="null" style="left:' + un.toFixed(2) + '%;opacity:.45"></i>' +
        '</div>' +
        '<div class="eta-vals"><b>' + (c.eta < 0.0005 ? '0.000' : c.eta.toFixed(3)) + '</b>' +
          ' <em>/</em> ' + (u.eta < 0.0005 ? '0.000' : u.eta.toFixed(3)) + '</div>' +
      '</div>';
    }).join('');
  }

  // ---------------- tables ----------------
  function renderTables() {
    var tb = document.querySelector('#baseTable tbody');
    tb.innerHTML = D.baselines.map(function (r) {
      var cls = r.rule.indexOf('SNHI') === 0 ? ' class="snhi"' : '';
      return '<tr' + cls + '><td>' + r.rule + '</td><td>' + r.norm + '</td><td>' + r.pos +
        '</td><td>' + r.between.toFixed(1) + '</td><td>' + r.within.toFixed(1) + '</td></tr>';
    }).join('');

    var bb = document.querySelector('#binTable tbody');
    bb.innerHTML = D.bins.map(function (r) {
      return '<tr><td>' + r.bw + '</td><td>' + r.between.toFixed(2) + '</td><td>' + r.within.toFixed(2) +
        '</td><td>' + (r.med == null ? '—' : r.med.toFixed(2)) +
        '</td><td>' + (r.max == null ? '—' : r.max.toFixed(2)) + '</td></tr>';
    }).join('');
  }

  // ---------------- toggle ----------------
  document.querySelectorAll('.align-seg button').forEach(function (b) {
    b.addEventListener('click', function () {
      state.metric = b.dataset.metric;
      document.querySelectorAll('.align-seg button').forEach(function (x) { x.classList.toggle('on', x === b); });
      document.getElementById('chartTitle').textContent = CFG[state.metric].title;
      document.getElementById('chartFoot').textContent = CFG[state.metric].foot;
      tip.style.display = 'none';
      draw();
    });
  });

  window.addEventListener('resize', draw);

  renderEta();
  renderTables();
  draw();
})();
