"""
HallucinationGuard-Env v4.2 — Production FastAPI Server

Endpoints:
  Standard   : POST /reset  POST /step  GET /state  GET /health
  OpenEnv    : GET /tasks  POST /grader  POST /baseline
  Extra      : GET /leaderboard  POST /leaderboard/submit  GET /datasets

"""

import sys, os, uuid, logging, dataclasses, enum, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from models import HallucinationAction, HallucinationObservation, HallucinationState
from environment import HallucinationEnvironment
from metrics import get_tracker

from tasks import (
    ALL_TASKS, get_task, task_id_for_difficulty, compute_task_score, ACTION_SCHEMA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# POLISHED DEMO PAGE
# ═══════════════════════════════════════════════════════════════════════════════

STUNNING_DOCS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HallucinationGuard-Env · OpenEnv</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%23080c14'/><text x='50' y='68' font-size='55' text-anchor='middle' fill='%23f59e0b' font-family='sans-serif' font-weight='bold'>H</text></svg>">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #080c14;
  --surface: #0d1321;
  --surface2: #111827;
  --border: #1e2d45;
  --border2: #243450;
  --text: #e2eaf5;
  --muted: #6b8aad;
  --amber: #f59e0b;
  --amber-dim: #78490a;
  --teal: #2dd4bf;
  --teal-dim: #0f4f49;
  --red: #f87171;
  --red-dim: #4c1515;
  --blue: #60a5fa;
  --blue-dim: #1e3a5f;
  --green: #4ade80;
  --green-dim: #14532d;
  --font: 'Space Grotesk', system-ui, sans-serif;
  --mono: 'Fira Code', monospace;
}
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── NOISE TEXTURE OVERLAY ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
  opacity: 0.4;
}

/* ── HERO ── */
.hero {
  position: relative;
  padding: 64px 40px 56px;
  background: linear-gradient(135deg, #0a1628 0%, #080c14 50%, #0a1420 100%);
  border-bottom: 1px solid var(--border);
  overflow: hidden;
}
.hero::after {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(245,158,11,0.07) 0%, transparent 70%);
  pointer-events: none;
}
.hero::before {
  content: '';
  position: absolute;
  bottom: -80px; left: 30%;
  width: 500px; height: 300px;
  background: radial-gradient(ellipse, rgba(45,212,191,0.05) 0%, transparent 70%);
  pointer-events: none;
}
.hero-inner { max-width: 1100px; margin: 0 auto; position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(245,158,11,0.1);
  border: 1px solid rgba(245,158,11,0.25);
  color: var(--amber);
  padding: 4px 12px; border-radius: 100px;
  font-size: 11px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; margin-bottom: 20px;
}
.hero-badge::before { content: '●'; font-size: 8px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.hero h1 {
  font-size: clamp(28px, 5vw, 48px);
  font-weight: 700;
  letter-spacing: -1px;
  line-height: 1.1;
  margin-bottom: 16px;
}
.hero h1 .accent { color: var(--amber); }
.hero h1 .accent2 { color: var(--teal); }
.hero-sub {
  font-size: 16px; color: var(--muted); max-width: 600px;
  margin-bottom: 36px; font-weight: 400; line-height: 1.7;
}
.stats-row {
  display: flex; gap: 32px; flex-wrap: wrap;
}
.stat-pill {
  display: flex; flex-direction: column;
  animation: fadeUp 0.6s ease both;
}
.stat-pill:nth-child(2) { animation-delay: 0.1s; }
.stat-pill:nth-child(3) { animation-delay: 0.2s; }
.stat-pill:nth-child(4) { animation-delay: 0.3s; }
@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
.stat-num {
  font-size: 32px; font-weight: 700;
  background: linear-gradient(135deg, var(--amber), var(--teal));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
}
.stat-lbl { font-size: 12px; color: var(--muted); margin-top: 4px; font-weight: 500; letter-spacing: 0.5px; }
.ver-chip {
  position: absolute; top: 24px; right: 40px;
  font-family: var(--mono); font-size: 11px;
  color: var(--muted); border: 1px solid var(--border2);
  padding: 4px 10px; border-radius: 4px;
  background: rgba(13,19,33,0.8);
}

/* ── NAV ── */
nav {
  position: sticky; top: 0; z-index: 100;
  background: rgba(8,12,20,0.95);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  padding: 0 40px;
}
.nav-inner {
  max-width: 1100px; margin: 0 auto;
  display: flex; gap: 0;
}
.tab {
  padding: 14px 20px;
  font-size: 13px; font-weight: 500;
  color: var(--muted); cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s; letter-spacing: 0.3px;
  user-select: none;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--amber); border-bottom-color: var(--amber); }

/* ── MAIN ── */
main { max-width: 1100px; margin: 0 auto; padding: 40px 40px 80px; }
.panel { display: none; animation: fadeIn 0.3s ease; }
.panel.active { display: block; }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

/* ── CARDS ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 16px;
  transition: border-color 0.2s;
}
.card:hover { border-color: var(--border2); }
.card h3 { font-size: 15px; font-weight: 600; margin-bottom: 10px; }
.card p { font-size: 14px; color: var(--muted); line-height: 1.7; }
.card p strong { color: var(--text); }

/* ── HOW IT WORKS ── */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
.step {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px 20px;
  text-align: center;
  position: relative;
}
.step-icon {
  width: 48px; height: 48px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  margin: 0 auto 14px;
}
.step:nth-child(1) .step-icon { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.2); }
.step:nth-child(2) .step-icon { background: rgba(45,212,191,0.12); border: 1px solid rgba(45,212,191,0.2); }
.step:nth-child(3) .step-icon { background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.2); }
.step-num {
  position: absolute; top: 12px; right: 12px;
  font-family: var(--mono); font-size: 10px;
  color: var(--border2); font-weight: 500;
}
.step h4 { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
.step p { font-size: 12px; color: var(--muted); line-height: 1.6; }
.step-arrow { display: flex; align-items: center; justify-content: center; color: var(--border2); font-size: 20px; }

/* ── TASK CARDS ── */
.task-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid;
  border-radius: 12px;
  padding: 20px 24px;
  margin-bottom: 14px;
  display: flex; gap: 16px; align-items: flex-start;
}
.task-card.beginner { border-left-color: var(--green); }
.task-card.intermediate { border-left-color: var(--blue); }
.task-card.advanced { border-left-color: var(--red); }
.task-icon { font-size: 28px; flex-shrink: 0; margin-top: 2px; }
.task-body { flex: 1; }
.task-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
.task-head h3 { font-size: 15px; font-weight: 600; }
.diff-badge {
  font-size: 10px; font-weight: 700; padding: 2px 8px;
  border-radius: 100px; text-transform: uppercase; letter-spacing: 1px;
}
.diff-badge.beginner { background: var(--green-dim); color: var(--green); }
.diff-badge.intermediate { background: var(--blue-dim); color: var(--blue); }
.diff-badge.advanced { background: var(--red-dim); color: var(--red); }
.data-count {
  font-family: var(--mono); font-size: 11px;
  color: var(--muted); margin-left: auto;
}
.task-body p { font-size: 13px; color: var(--muted); line-height: 1.6; }
.dataset-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.ds-chip {
  font-size: 10px; font-family: var(--mono);
  padding: 2px 8px; border-radius: 4px;
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--muted);
}

/* ── API TABLE ── */
.api-table-wrap { overflow: auto; border-radius: 10px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  text-align: left; padding: 12px 16px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  font-size: 11px; color: var(--muted);
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;
}
td { padding: 12px 16px; border-bottom: 1px solid var(--border); color: var(--text); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.015); }
.method {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  padding: 3px 8px; border-radius: 4px; letter-spacing: 0.5px;
}
.method.get { background: var(--green-dim); color: var(--green); }
.method.post { background: var(--blue-dim); color: var(--blue); }
.endpoint { font-family: var(--mono); font-size: 12px; color: var(--amber); }
.td-desc { color: var(--muted); font-size: 12px; }

/* ── CODE BLOCK ── */
.code-block {
  background: #050810;
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 16px;
}
.code-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 16px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
}
.code-lang { font-size: 11px; color: var(--muted); font-family: var(--mono); }
.copy-btn {
  font-size: 11px; color: var(--muted);
  background: none; border: 1px solid var(--border);
  padding: 3px 10px; border-radius: 4px; cursor: pointer;
  font-family: var(--font); transition: all 0.15s;
}
.copy-btn:hover { color: var(--text); border-color: var(--border2); }
.code-body {
  padding: 20px;
  font-family: var(--mono); font-size: 12px;
  line-height: 1.8; color: #c9d4e8;
  overflow-x: auto; white-space: pre;
}
.code-body .cm { color: #4a6a8a; }
.code-body .kw { color: var(--amber); }
.code-body .st { color: var(--teal); }
.code-body .fn { color: var(--blue); }
.code-body .hl { color: var(--green); }

/* ── PLAYGROUND ── */
.pg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.pg-label {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 8px; display: block;
}
.pg-select, .pg-input, .pg-textarea {
  width: 100%; padding: 10px 14px;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: 8px; font-size: 13px; font-family: var(--font);
  color: var(--text); outline: none; transition: border-color 0.2s;
  appearance: none;
}
.pg-select:focus, .pg-input:focus, .pg-textarea:focus {
  border-color: var(--amber);
}
.pg-textarea { min-height: 80px; resize: vertical; line-height: 1.5; }
.pg-select option { background: var(--surface2); }
.slider-wrap { margin-top: 4px; }
.slider-row { display: flex; align-items: center; gap: 12px; }
input[type=range] {
  flex: 1; height: 4px; border-radius: 2px;
  background: var(--border2); outline: none; cursor: pointer;
  accent-color: var(--amber);
}
.slider-val {
  font-family: var(--mono); font-size: 13px; font-weight: 600;
  color: var(--amber); min-width: 30px; text-align: right;
}
.btn-row { display: flex; gap: 10px; margin-top: 16px; }
.btn {
  padding: 10px 20px; border-radius: 8px;
  font-size: 13px; font-weight: 600; cursor: pointer;
  transition: all 0.15s; border: 1px solid;
  font-family: var(--font); letter-spacing: 0.3px;
}
.btn-primary {
  background: var(--amber); color: #0a0a0a;
  border-color: var(--amber);
}
.btn-primary:hover { background: #fbbf24; }
.btn-primary:active { transform: scale(0.97); }
.btn-secondary {
  background: transparent; color: var(--text);
  border-color: var(--border2);
}
.btn-secondary:hover { background: var(--surface2); border-color: var(--muted); }
.btn-secondary:disabled { opacity: 0.4; cursor: not-allowed; }
.input-group { margin-bottom: 14px; }

/* ── CONTEXT DISPLAY ── */
.context-box {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px; padding: 16px;
  font-size: 13px; line-height: 1.7;
  color: var(--muted); min-height: 80px;
  max-height: 180px; overflow-y: auto;
  margin-bottom: 4px;
}
.context-box .q-highlight {
  background: rgba(245,158,11,0.15);
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: 4px; padding: 2px 6px;
  color: var(--text); font-weight: 500;
}
.empty-hint { color: var(--border2); font-style: italic; font-size: 12px; }

/* ── EPISODE PROGRESS ── */
.ep-progress { margin-bottom: 16px; }
.ep-bar-bg {
  height: 4px; background: var(--border);
  border-radius: 2px; overflow: hidden; margin-top: 6px;
}
.ep-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--amber), var(--teal));
  border-radius: 2px;
  transition: width 0.4s ease;
}
.ep-meta { display: flex; justify-content: space-between; align-items: center; }
.ep-step { font-size: 11px; color: var(--muted); font-family: var(--mono); }
.halluc-badge {
  display: none;
  font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
  padding: 3px 10px; border-radius: 100px;
}
.halluc-badge.show { display: inline-block; }
.halluc-badge.yes { background: var(--red-dim); color: var(--red); border: 1px solid rgba(248,113,113,0.3); }
.halluc-badge.no { background: var(--green-dim); color: var(--green); border: 1px solid rgba(74,222,128,0.3); }

/* ── REWARD BREAKDOWN ── */
.reward-section { margin-top: 16px; }
.reward-title {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;
  display: flex; justify-content: space-between; align-items: center;
}
.total-reward-val {
  font-family: var(--mono); font-size: 16px;
  font-weight: 700; color: var(--amber);
}
.reward-bars { display: flex; flex-direction: column; gap: 8px; }
.reward-bar-row { display: flex; align-items: center; gap: 10px; }
.rb-label { font-size: 11px; color: var(--muted); width: 130px; flex-shrink: 0; font-family: var(--mono); }
.rb-track {
  flex: 1; height: 6px; background: var(--border);
  border-radius: 3px; overflow: hidden;
}
.rb-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.5s cubic-bezier(.4,0,.2,1);
  width: 0%;
}
.rb-val { font-family: var(--mono); font-size: 11px; color: var(--text); width: 36px; text-align: right; flex-shrink: 0; }

/* ── RAW RESPONSE ── */
.raw-toggle {
  font-size: 11px; color: var(--muted); cursor: pointer;
  display: flex; align-items: center; gap: 6px; margin-top: 12px;
  user-select: none;
}
.raw-toggle:hover { color: var(--text); }
.raw-box {
  display: none; margin-top: 8px;
  background: #050810; border: 1px solid var(--border);
  border-radius: 8px; padding: 14px;
  font-family: var(--mono); font-size: 11px;
  color: var(--muted); white-space: pre-wrap;
  max-height: 200px; overflow-y: auto; line-height: 1.6;
}
.raw-box.open { display: block; }

/* ── SECTION HEADING ── */
.section-head { margin-bottom: 20px; }
.section-head h2 { font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
.section-head p { font-size: 13px; color: var(--muted); margin-top: 4px; }

/* ── REWARD COLORS ── */
.rc-0 { background: var(--green); }
.rc-1 { background: var(--teal); }
.rc-2 { background: var(--blue); }
.rc-3 { background: var(--amber); }
.rc-4 { background: #a78bfa; }
.rc-5 { background: #fb923c; }
.rc-6 { background: #34d399; }
.rc-7 { background: #f472b6; }
.rc-8 { background: #e879f9; }

/* ── STATUS INDICATOR ── */
.status-dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; background: var(--muted);
}
.status-dot.ready { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
.status-dot.done { background: var(--teal); box-shadow: 0 0 6px var(--teal); }

/* ── RESPONSIVE ── */
@media(max-width: 768px) {
  .hero { padding: 48px 20px 40px; }
  .ver-chip { display: none; }
  main { padding: 24px 20px 60px; }
  nav { padding: 0 20px; }
  .stats-row { gap: 20px; }
  .steps { grid-template-columns: 1fr; }
  .step-arrow { display: none; }
  .pg-grid { grid-template-columns: 1fr; }
  .tab { padding: 12px 14px; font-size: 12px; }
}
</style>
</head>
<body>

<!-- ══ HERO ══ -->
<section class="hero">
  <div class="hero-inner">
    <div class="hero-badge">OpenEnv · RL Environment</div>
    <div class="ver-chip">v4.2.0</div>
    <h1>
      <span class="accent">Hallucination</span><span class="accent2">Guard</span>‑Env
    </h1>
    <p class="hero-sub">
      Train AI models to answer <strong>only from verified context</strong> — with a 9-component reward system that penalizes fabrication and rewards factual grounding, citation accuracy, and calibrated confidence.
    </p>
    <div class="stats-row">
      <div class="stat-pill">
        <span class="stat-num" data-target="1000000" data-suffix="M+">1M+</span>
        <span class="stat-lbl">Examples</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">38</span>
        <span class="stat-lbl">Datasets</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">3</span>
        <span class="stat-lbl">Task Tiers</span>
      </div>
      <div class="stat-pill">
        <span class="stat-num">9</span>
        <span class="stat-lbl">Reward Components</span>
      </div>
    </div>
  </div>
</section>

<!-- ══ NAV ══ -->
<nav>
  <div class="nav-inner">
    <div class="tab active" onclick="showTab('overview',this)">Overview</div>
    <div class="tab" onclick="showTab('tasks',this)">Tasks</div>
    <div class="tab" onclick="showTab('api',this)">API Reference</div>
    <div class="tab" onclick="showTab('playground',this)">Playground</div>
  </div>
</nav>

<!-- ══ MAIN ══ -->
<main>

<!-- ─ OVERVIEW ─ -->
<div id="overview" class="panel active">
  <div class="section-head">
    <h2>How it works</h2>
    <p>Three primitives. Nine reward signals. One goal: no hallucinations.</p>
  </div>
  <div class="steps">
    <div class="step">
      <span class="step-num">01</span>
      <div class="step-icon">🔄</div>
      <h4>reset()</h4>
      <p>Sample a question + context document from one of 38 curated datasets, stratified by difficulty tier.</p>
    </div>
    <div class="step">
      <span class="step-num">02</span>
      <div class="step-icon">📤</div>
      <h4>step(answer)</h4>
      <p>Submit your answer with confidence and a source quote. Receive a dense reward signal across all 9 components.</p>
    </div>
    <div class="step">
      <span class="step-num">03</span>
      <div class="step-icon">📊</div>
      <h4>grade()</h4>
      <p>Aggregate episode rewards into a task score. Track accuracy, hallucination rate, and skill rating over time.</p>
    </div>
  </div>

  <div class="card">
    <h3>9-Component Reward System</h3>
    <p>Every answer is graded on <strong>factual correctness</strong>, <strong>source grounding</strong>, <strong>citation accuracy</strong>, <strong>confidence calibration</strong>, <strong>semantic consistency</strong>, <strong>hallucination detection</strong>, <strong>ROUGE-L</strong>, <strong>BERTScore</strong>, and <strong>AlignScore</strong>. Each component is weighted and combined into a single scalar reward in <strong>[0, 1]</strong>. Confident wrong answers are penalized harder than uncertain ones.</p>
  </div>
  <div class="card">
    <h3>Curriculum Progression</h3>
    <p>Episodes advance from <strong>Beginner</strong> (single-hop factual QA with unambiguous ground-truth) through <strong>Intermediate</strong> (multi-hop synthesis across multiple context sentences) to <strong>Advanced</strong> (adversarial prompts where confident refusals score best). The environment tracks a live <strong>skill rating</strong> and adjusts difficulty sampling accordingly.</p>
  </div>
</div>

<!-- ─ TASKS ─ -->
<div id="tasks" class="panel">
  <div class="section-head">
    <h2>Task Tiers</h2>
    <p>Three progressively harder tasks drawn from 38 datasets with 1M+ examples.</p>
  </div>
  <div class="task-card beginner">
    <div class="task-icon">🟢</div>
    <div class="task-body">
      <div class="task-head">
        <h3>Factual Grounding</h3>
        <span class="diff-badge beginner">Beginner</span>
        <span class="data-count">~450K examples</span>
      </div>
      <p>Answer straightforward factual questions from a short context passage. Single-hop retrieval with unambiguous ground truth. The grader rewards precise citation and heavily penalizes adding information not found in the context.</p>
      <div class="dataset-chips">
        <span class="ds-chip">SQuAD</span>
        <span class="ds-chip">BoolQ</span>
        <span class="ds-chip">OpenBookQA</span>
        <span class="ds-chip">ARC</span>
        <span class="ds-chip">TriviaQA</span>
        <span class="ds-chip">+8 more</span>
      </div>
    </div>
  </div>
  <div class="task-card intermediate">
    <div class="task-icon">🔵</div>
    <div class="task-body">
      <div class="task-head">
        <h3>Multi-Hop Synthesis</h3>
        <span class="diff-badge intermediate">Intermediate</span>
        <span class="data-count">~380K examples</span>
      </div>
      <p>Synthesize evidence from multiple context sentences to reach an answer. Requires connecting disparate facts without fabricating bridge claims. AlignScore and BERTScore are weighted more heavily at this tier.</p>
      <div class="dataset-chips">
        <span class="ds-chip">HotpotQA</span>
        <span class="ds-chip">CoQA</span>
        <span class="ds-chip">NQ-Open</span>
        <span class="ds-chip">MS-MARCO</span>
        <span class="ds-chip">MuSiQue</span>
        <span class="ds-chip">+7 more</span>
      </div>
    </div>
  </div>
  <div class="task-card advanced">
    <div class="task-icon">🔴</div>
    <div class="task-body">
      <div class="task-head">
        <h3>Adversarial Resistance</h3>
        <span class="diff-badge advanced">Advanced</span>
        <span class="data-count">~210K examples</span>
      </div>
      <p>Resist adversarial prompts designed to elicit hallucinations. Many questions are deliberately unanswerable — confident refusals with low confidence score better than fabricated plausible-sounding answers.</p>
      <div class="dataset-chips">
        <span class="ds-chip">HaluEval</span>
        <span class="ds-chip">TruthfulQA</span>
        <span class="ds-chip">FEVER</span>
        <span class="ds-chip">Climate-FEVER</span>
        <span class="ds-chip">WittyQA</span>
        <span class="ds-chip">+6 more</span>
      </div>
    </div>
  </div>
</div>

<!-- ─ API ─ -->
<div id="api" class="panel">
  <div class="section-head">
    <h2>API Reference</h2>
    <p>RESTful JSON API. All endpoints accept and return <code style="font-family:var(--mono);font-size:12px;color:var(--amber)">application/json</code>. No auth required.</p>
  </div>
  <div class="api-table-wrap">
    <table>
      <thead>
        <tr>
          <th>Method</th>
          <th>Endpoint</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><span class="method post">POST</span></td><td class="endpoint">/reset</td><td class="td-desc">Start episode — returns question, context, difficulty, episode_id</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="endpoint">/step</td><td class="td-desc">Submit answer with confidence + source_quote, receive reward breakdown</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="endpoint">/state</td><td class="td-desc">Current episode metadata — accuracy, hallucination_rate, skill_rating</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="endpoint">/tasks</td><td class="td-desc">List all 3 tasks with action schema</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="endpoint">/grader</td><td class="td-desc">Score a completed episode (0.0 – 1.0) from rewards + infos</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="endpoint">/baseline</td><td class="td-desc">Run heuristic baseline across all 3 tasks</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="endpoint">/metadata</td><td class="td-desc">Environment name, version, license</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="endpoint">/schema</td><td class="td-desc">Full JSON schemas for action, observation, state</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="endpoint">/health</td><td class="td-desc">Health check — returns {"status":"healthy"}</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="endpoint">/mcp</td><td class="td-desc">JSON-RPC 2.0 tool discovery for MCP clients</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="endpoint">/leaderboard</td><td class="td-desc">Ranked leaderboard by avg_reward</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="endpoint">/leaderboard/submit</td><td class="td-desc">Submit model results for ranking</td></tr>
      </tbody>
    </table>
  </div>

  <div style="margin-top:28px">
    <div class="section-head"><h2>Quick Start</h2><p>Three commands to run your first episode.</p></div>
    <div class="code-block">
      <div class="code-header">
        <span class="code-lang">bash</span>
        <button class="copy-btn" onclick="copyCode(this,'install')">Copy</button>
      </div>
      <div class="code-body" id="install"><span class="cm"># Install and launch</span>
<span class="kw">pip</span> install -e .
<span class="kw">uvicorn</span> server.app:app --port 7860

<span class="cm"># Run heuristic baseline</span>
<span class="kw">python</span> inference.py --heuristic --env-url http://localhost:7860</div>
    </div>
    <div class="code-block">
      <div class="code-header">
        <span class="code-lang">python</span>
        <button class="copy-btn" onclick="copyCode(this,'pycode')">Copy</button>
      </div>
      <div class="code-body" id="pycode"><span class="kw">import</span> requests

BASE = <span class="st">"http://localhost:7860"</span>

<span class="cm"># 1. Reset — get a question + context</span>
obs = requests.<span class="fn">post</span>(<span class="st">f"{BASE}/reset"</span>, json={<span class="st">"difficulty"</span>: <span class="st">"beginner"</span>}).json()
session_id = obs[<span class="st">"session_id"</span>]
<span class="fn">print</span>(obs[<span class="st">"question"</span>])

<span class="cm"># 2. Step — submit your answer</span>
result = requests.<span class="fn">post</span>(<span class="st">f"{BASE}/step"</span>, json={
    <span class="st">"answer"</span>:       <span class="st">"Based on the context, ..."</span>,
    <span class="st">"confidence"</span>:   <span class="hl">0.85</span>,
    <span class="st">"source_quote"</span>: <span class="st">"verbatim text from context"</span>,
    <span class="st">"session_id"</span>:   session_id,
}).json()

<span class="fn">print</span>(result[<span class="st">"reward"</span>])            <span class="cm"># scalar in [0, 1]</span>
<span class="fn">print</span>(result[<span class="st">"is_hallucination"</span>])   <span class="cm"># bool</span></div>
    </div>
  </div>
</div>

<!-- ─ PLAYGROUND ─ -->
<div id="playground" class="panel">
  <div class="section-head">
    <h2>Interactive Playground</h2>
    <p>Reset an episode, read the context, craft your answer, and see the live reward breakdown.</p>
  </div>
  <div class="pg-grid">

    <!-- LEFT: Controls -->
    <div>
      <div class="card">
        <div class="ep-progress">
          <div class="ep-meta">
            <span class="ep-step" id="ep-step-label">No episode active</span>
            <span class="halluc-badge" id="halluc-badge"></span>
          </div>
          <div class="ep-bar-bg"><div class="ep-bar-fill" id="ep-bar" style="width:0%"></div></div>
        </div>

        <div class="input-group">
          <label class="pg-label">Difficulty</label>
          <select class="pg-select" id="difficulty">
            <option value="beginner">🟢  Beginner — Factual Grounding</option>
            <option value="intermediate">🔵  Intermediate — Multi-Hop Synthesis</option>
            <option value="advanced">🔴  Advanced — Adversarial Resistance</option>
          </select>
        </div>

        <div class="input-group">
          <label class="pg-label">Question &amp; Context</label>
          <div class="context-box" id="ctx-box">
            <span class="empty-hint">Click Reset to load a question and context...</span>
          </div>
        </div>

        <div class="input-group">
          <label class="pg-label">Your Answer</label>
          <textarea class="pg-textarea" id="answer" placeholder="Write an answer derived only from the context above..."></textarea>
        </div>

        <div class="input-group">
          <label class="pg-label">Confidence</label>
          <div class="slider-wrap">
            <div class="slider-row">
              <input type="range" min="0" max="1" step="0.05" value="0.7" id="confidence"
                oninput="document.getElementById('conf-val').textContent=parseFloat(this.value).toFixed(2)">
              <span class="slider-val" id="conf-val">0.70</span>
            </div>
          </div>
        </div>

        <div class="input-group">
          <label class="pg-label">Source Quote <span style="color:var(--border2);font-weight:400;text-transform:none;letter-spacing:0">(verbatim from context)</span></label>
          <input class="pg-input" id="source_quote" type="text" placeholder="Paste a verbatim phrase from the context...">
        </div>

        <div class="btn-row">
          <button class="btn btn-primary" onclick="doReset()">⟳ Reset</button>
          <button class="btn btn-secondary" id="step-btn" onclick="doStep()" disabled>→ Step</button>
        </div>
      </div>
    </div>

    <!-- RIGHT: Results -->
    <div>
      <div class="card" id="reward-card">
        <div class="reward-title">
          <span>Reward Breakdown</span>
          <span class="total-reward-val" id="total-reward">—</span>
        </div>
        <div class="reward-bars" id="reward-bars">
          <!-- placeholder bars -->
          <div style="text-align:center;padding:20px 0;color:var(--border2);font-size:13px;">
            Submit an answer to see the 9-component reward breakdown
          </div>
        </div>
        <div class="raw-toggle" onclick="toggleRaw()">
          <span id="raw-arrow">▶</span> Raw JSON response
        </div>
        <div class="raw-box" id="raw-box"></div>
      </div>

      <div class="card">
        <div class="reward-title" style="margin-bottom:14px">
          <span>Observation</span>
          <span style="display:flex;align-items:center;gap:6px">
            <span class="status-dot" id="status-dot"></span>
            <span id="status-text" style="font-size:11px;color:var(--muted);font-weight:500"></span>
          </span>
        </div>
        <div class="raw-box open" id="obs-box" style="max-height:220px">Click Reset to start an episode.</div>
      </div>
    </div>

  </div>
</div>

</main>

<!-- ══ FOOTER ══ -->
<footer style="text-align:center;padding:32px 40px 24px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;">
  HallucinationGuard-Env v4.2.0 &middot; OpenEnv &middot; <a href="/swagger" style="color:var(--amber);text-decoration:none">Swagger Docs</a> &middot; <a href="/redoc" style="color:var(--amber);text-decoration:none">ReDoc</a>
</footer>

<script>
let sessionId = null;
let stepCount = 0;
const MAX_STEPS = 10;

const REWARD_KEYS = [
  {key:'correctness',           label:'Factual Correctness',  css:'rc-0'},
  {key:'grounding',             label:'Source Grounding',     css:'rc-1'},
  {key:'citation',              label:'Citation Accuracy',    css:'rc-2'},
  {key:'calibration',          label:'Confidence Calibr.',   css:'rc-3'},
  {key:'consistency',           label:'Semantic Consistency', css:'rc-4'},
  {key:'halluc_detect',        label:'Hallucination Detect.', css:'rc-5'},
  {key:'rouge_l',               label:'ROUGE-L',             css:'rc-6'},
  {key:'bert_score',            label:'BERTScore',            css:'rc-7'},
  {key:'align_score',           label:'AlignScore',           css:'rc-8'},
  // Also accept alternate key names from the grader
  {key:'factual_correctness',   label:'Factual Correctness',  css:'rc-0'},
  {key:'source_grounding',      label:'Source Grounding',     css:'rc-1'},
  {key:'citation_accuracy',     label:'Citation Accuracy',    css:'rc-2'},
  {key:'confidence_calibration', label:'Confidence Calibr.',   css:'rc-3'},
  {key:'semantic_consistency',  label:'Semantic Consistency', css:'rc-4'},
  {key:'hallucination_penalty', label:'Hallucination Detect.', css:'rc-5'},
  {key:'rouge_score',           label:'ROUGE-L',              css:'rc-6'},
  {key:'bertscore',             label:'BERTScore',            css:'rc-7'},
  {key:'alignscore',            label:'AlignScore',           css:'rc-8'},
];

function showTab(id, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById(id).classList.add('active');
  if (id === 'playground') {
    document.getElementById('playground').scrollIntoView({behavior:'smooth', block:'start'});
  }
}

function setStatus(state) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  dot.className = 'status-dot ' + state;
  txt.textContent = state === 'ready' ? 'Ready' : state === 'done' ? 'Episode done' : '';
}

function updateProgress() {
  const pct = sessionId ? (stepCount / MAX_STEPS) * 100 : 0;
  document.getElementById('ep-bar').style.width = pct + '%';
  document.getElementById('ep-step-label').textContent = sessionId
    ? `Step ${stepCount} / ${MAX_STEPS}`
    : 'No episode active';
}

function renderContext(question, context) {
  const box = document.getElementById('ctx-box');
  box.innerHTML = `<div style="margin-bottom:10px"><span style="font-size:10px;font-weight:700;letter-spacing:1px;color:var(--amber);text-transform:uppercase">Question</span><br><span class="q-highlight">${escHtml(question)}</span></div><div><span style="font-size:10px;font-weight:700;letter-spacing:1px;color:var(--muted);text-transform:uppercase">Context</span><br><span style="font-size:12px;color:var(--muted)">${escHtml(context)}</span></div>`;
}

function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderRewards(data) {
  const container = document.getElementById('reward-bars');
  const reward = data.reward != null ? data.reward : data.total_reward;
  const totalEl = document.getElementById('total-reward');
  if (reward != null) {
    const r = parseFloat(reward);
    totalEl.textContent = (r >= 0 ? '+' : '') + r.toFixed(3);
    totalEl.style.color = r >= 0 ? 'var(--teal)' : 'var(--red)';
  }

  const info = data.info || data.reward_breakdown || data.breakdown || data.metadata || {};
  let html = '';
  let foundAny = false;
  const seen = new Set();
  REWARD_KEYS.forEach(({key, label, css}) => {
    if (seen.has(label)) return;
    let v = info[key] != null ? info[key] : (data[key] != null ? data[key] : null);
    if (v == null) return;
    seen.add(label);
    foundAny = true;
    const pct = Math.min(100, Math.max(0, Math.round(Math.abs(parseFloat(v)) * 100)));
    const display = parseFloat(v).toFixed(3);
    html += `<div class="reward-bar-row">
      <span class="rb-label">${label}</span>
      <div class="rb-track"><div class="rb-fill ${css}" style="width:${pct}%"></div></div>
      <span class="rb-val">${display}</span>
    </div>`;
  });

  if (!foundAny && reward != null) {
    const pct = Math.min(100, Math.max(0, Math.round(parseFloat(reward)*100)));
    html = `<div class="reward-bar-row">
      <span class="rb-label">total_reward</span>
      <div class="rb-track"><div class="rb-fill rc-0" style="width:${pct}%"></div></div>
      <span class="rb-val">${parseFloat(reward).toFixed(3)}</span>
    </div>`;
  }

  container.innerHTML = html || '<div style="color:var(--border2);font-size:12px;text-align:center;padding:12px">No breakdown data in response</div>';

  // hallucination badge
  const badge = document.getElementById('halluc-badge');
  if (data.is_hallucination != null) {
    badge.className = 'halluc-badge show ' + (data.is_hallucination ? 'yes' : 'no');
    badge.textContent = data.is_hallucination ? '⚠ Hallucination' : '✓ Grounded';
  }
}

async function doReset() {
  const diff = document.getElementById('difficulty').value;
  const resetBtn = document.querySelector('.btn-primary');
  resetBtn.disabled = true; resetBtn.textContent = 'Loading...';
  document.getElementById('ctx-box').innerHTML = '<span class="empty-hint">Loading...</span>';
  document.getElementById('obs-box').textContent = 'Loading...';
  document.getElementById('step-btn').disabled = true;
  setStatus('');
  try {
    const r = await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({difficulty: diff, seed: Math.floor(Math.random()*9999)})
    });
    const data = await r.json();
    sessionId = data.session_id || null;
    stepCount = 0;
    updateProgress();
    renderContext(data.question || '(no question)', data.context || '(no context)');
    document.getElementById('obs-box').textContent = JSON.stringify(data, null, 2);
    document.getElementById('step-btn').disabled = false;
    document.getElementById('reward-bars').innerHTML = '<div style="text-align:center;padding:20px 0;color:var(--border2);font-size:13px;">Submit an answer to see the 9-component reward breakdown</div>';
    document.getElementById('total-reward').textContent = '—';
    document.getElementById('total-reward').style.color = 'var(--amber)';
    document.getElementById('halluc-badge').className = 'halluc-badge';
    setStatus('ready');
  } catch(e) {
    document.getElementById('ctx-box').innerHTML = '<span style="color:var(--red)">Error: ' + escHtml(e.message) + '</span>';
    setStatus('');
  } finally {
    resetBtn.disabled = false; resetBtn.textContent = '⟳ Reset';
  }
}

async function doStep() {
  if (!sessionId) { alert('Reset first!'); return; }
  const stepBtn = document.getElementById('step-btn');
  const body = {
    answer: document.getElementById('answer').value,
    confidence: parseFloat(document.getElementById('confidence').value) || 0.5,
    source_quote: document.getElementById('source_quote').value,
    session_id: sessionId,
  };
  stepBtn.disabled = true; stepBtn.textContent = 'Submitting...';
  try {
    const r = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await r.json();
    stepCount++;
    updateProgress();
    renderRewards(data);
    document.getElementById('raw-box').textContent = JSON.stringify(data, null, 2);
    if (data.done) {
      sessionId = null;
      stepBtn.textContent = '→ Step';
      document.getElementById('ctx-box').innerHTML = '<span class="empty-hint">Episode complete. Click Reset for a new episode.</span>';
      setStatus('done');
    } else {
      stepBtn.disabled = false; stepBtn.textContent = '→ Step';
      setStatus('ready');
      if (data.question) renderContext(data.question, data.context || '');
    }
  } catch(e) {
    document.getElementById('raw-box').textContent = 'Error: ' + e.message;
    stepBtn.disabled = false; stepBtn.textContent = '→ Step';
    setStatus('');
  }
}

function toggleRaw() {
  const box = document.getElementById('raw-box');
  const arr = document.getElementById('raw-arrow');
  box.classList.toggle('open');
  arr.textContent = box.classList.contains('open') ? '▼' : '▶';
}

function copyCode(btn, id) {
  const text = document.getElementById(id).textContent;
  const doCopy = (txt) => {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(txt);
    }
    const ta = document.createElement('textarea');
    ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
    return Promise.resolve();
  };
  doCopy(text).then(() => {
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1800);
  }).catch(() => {
    btn.textContent = 'Failed'; setTimeout(() => btn.textContent = 'Copy', 1800);
  });
}
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP — session-isolated environments for thread safety
# ═══════════════════════════════════════════════════════════════════════════════

_default_env: Optional[HallucinationEnvironment] = None
_env_loading = False
_env_lock = threading.Lock()

def _get_default_env() -> HallucinationEnvironment:
    """Get or create the shared dataset-loader environment (used only for dataset access)."""
    global _default_env, _env_loading
    if _default_env is not None:
        return _default_env
    with _env_lock:
        if _default_env is not None:
            return _default_env
        _env_loading = True
        try:
            logger.info("Creating HallucinationEnvironment (dataset loader)...")
            _default_env = HallucinationEnvironment()
            logger.info(f"Environment ready — {_default_env.dataset_loader.get_total_examples():,} examples loaded.")
            return _default_env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            from dataset_loader import DatasetLoader
            class MinimalEnv:
                def __init__(self):
                    self.dataset_loader = DatasetLoader()
                    self.dataset_loader.examples = []
                def reset(self, **kwargs):
                    return type('Obs', (), {'question': 'Placeholder', 'context': 'Context', 'reward': 0.0, 'done': False, 'info': {}})()
                def step(self, action):
                    return type('Obs', (), {'reward': 0.0, 'done': False, 'is_hallucination': False, 'info': {}})()
                def state(self): return {}
                def close(self): pass
            _default_env = MinimalEnv()
            return _default_env
        finally:
            _env_loading = False


def _create_session_env(session_id: str) -> HallucinationEnvironment:
    """Create a fresh per-session environment that shares the dataset loader
    (expensive to load) but has its own episode state (safe for concurrent use)."""
    loader_env = _get_default_env()
    # Pass the shared loader directly into __init__ so we skip the expensive
    # DatasetLoader() construction and dataset loading that would otherwise
    # happen inside HallucinationEnvironment.__init__
    env = HallucinationEnvironment(session_id=session_id, dataset_loader=loader_env.dataset_loader)
    return env


_sessions: Dict[str, HallucinationEnvironment] = {}
_session_lock = threading.Lock()


def _get_session(session_id: str) -> Optional[HallucinationEnvironment]:
    """Retrieve an existing session environment."""
    with _session_lock:
        return _sessions.get(session_id)


def _cleanup_session(session_id: str):
    """Remove and clean up a session environment."""
    with _session_lock:
        env = _sessions.pop(session_id, None)
    if env:
        try: env.close()
        except: pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _default_env

    def preload_models():
        try:
            logger.info("Preloading ML models...")
            import transformers
            transformers.logging.set_verbosity_error()
            from sentence_transformers import SentenceTransformer, CrossEncoder
            SentenceTransformer('all-MiniLM-L6-v2')
            CrossEncoder('cross-encoder/nli-deberta-v3-small')
            from rouge_score import rouge_scorer
            rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            try:
                from grader import _get_bert_scorer
                _get_bert_scorer()
            except: pass
            logger.info("All ML models preloaded!")
        except Exception as e:
            logger.error(f"Model preload failed: {e}")

    threading.Thread(target=preload_models, daemon=True).start()

    def background_load():
        try:
            logger.info("Background dataset loading...")
            env = _get_default_env()
            logger.info(f"Loaded {env.dataset_loader.get_total_examples():,} examples.")
        except Exception as e:
            logger.error(f"Background loading failed: {e}")

    threading.Thread(target=background_load, daemon=True).start()
    yield
    if _default_env:
        try: _default_env.close()
        except: pass

app = FastAPI(
    lifespan=lifespan,
    title="HallucinationGuard-Env",
    version="4.2.0",
    docs_url="/swagger",
    redoc_url="/redoc",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

import json as _json
_LEADERBOARD_FILE = "/tmp/hallucination_guard_leaderboard.json"

def _load_leaderboard():
    if os.path.exists(_LEADERBOARD_FILE):
        try:
            with open(_LEADERBOARD_FILE, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    return {}

def _save_leaderboard(lb):
    try:
        with open(_LEADERBOARD_FILE, "w", encoding="utf-8") as f:
            _json.dump(lb, f, indent=2)
    except Exception:
        pass

_leaderboard: Dict[str, Dict[str, Any]] = _load_leaderboard()

def _safe_dict(obj):
    if hasattr(obj, 'model_dump'): return _safe_dict(obj.model_dump())
    if hasattr(obj, 'dict'): return _safe_dict(obj.dict())
    if dataclasses.is_dataclass(obj): return {f.name: _safe_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum): return obj.value
    if isinstance(obj, dict): return {k: _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_safe_dict(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))): return obj
    return str(obj)

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root(): return STUNNING_DOCS_HTML

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # SVG favicon served as data-uri in the HTML; return 204 so browsers stop logging 404s
    return Response(status_code=204)

@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def docs(): return STUNNING_DOCS_HTML

@app.post("/reset", tags=["Environment"])
async def reset(body: Dict[str, Any] = {}):
    try:
        # Create a per-session environment for thread safety
        session_id = body.get("session_id") or f"ses_{uuid.uuid4().hex[:8]}"
        env = _create_session_env(session_id)
        obs = env.reset(**{k: v for k, v in body.items() if k in ("seed", "episode_id", "difficulty")})
        # Store the episode_id -> session mapping so /step can find this env
        episode_id = getattr(obs, 'episode_id', None) or body.get("episode_id") or session_id
        with _session_lock:
            _sessions[episode_id] = env
            _sessions[session_id] = env
        result = _safe_dict(obs)
        result["session_id"] = session_id
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        logger.error(f"Reset error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.post("/step", tags=["Environment"])
async def step(action_data: Dict[str, Any]):
    try:
        # Look up session by episode_id or session_id for thread safety
        session_id = action_data.pop("session_id", None) or action_data.pop("episode_id", None)
        env = _get_session(session_id) if session_id else None
        if env is None:
            # Fallback: use default env (single-user mode)
            env = _get_default_env()
        valid = set(HallucinationAction.model_fields.keys()) if hasattr(HallucinationAction, 'model_fields') else set(HallucinationAction.__fields__.keys())
        action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
        result = _safe_dict(env.step(action))
        # If episode is done, clean up session
        if result.get("done", False) and session_id:
            _cleanup_session(session_id)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        logger.error(f"Step error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.get("/state", tags=["Environment"])
async def get_state(session_id: Optional[str] = None):
    try:
        env = _get_session(session_id) if session_id else _get_default_env()
        return JSONResponse(content=_safe_dict(env.state()))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/tasks", tags=["OpenEnv"])
async def list_tasks():
    ordered = ["task_1_factual_grounding", "task_2_multi_hop_synthesis", "task_3_adversarial_resistance"]
    return {"tasks": [ALL_TASKS[t].to_dict() for t in ordered if t in ALL_TASKS], "action_schema": ACTION_SCHEMA}

@app.post("/grader", tags=["OpenEnv"])
async def grade_episode(body: Dict[str, Any]):
    task_id = body.get("task_id")
    if not task_id: raise HTTPException(422, "'task_id' required")
    task = get_task(task_id)
    if not task: raise HTTPException(404, f"task_id '{task_id}' not found")
    rewards, infos = body.get("step_rewards", []), body.get("step_infos", [])
    if not infos and rewards: return {"task_id": task_id, "score": round(sum(rewards)/len(rewards), 4)}
    return compute_task_score(task, rewards, infos)

@app.post("/baseline", tags=["OpenEnv"])
async def run_baseline(body: Dict[str, Any] = {}):
    steps = max(3, min(10, int(body.get("steps_per_task", 5))))
    seed = int(body.get("seed", 42))
    results = []
    for task_id, diff in [("task_1_factual_grounding","beginner"),("task_2_multi_hop_synthesis","intermediate"),("task_3_adversarial_resistance","advanced")]:
        task = get_task(task_id)
        if not task: continue
        sid = f"bl_{task_id}_{seed}"
        # Use session-based env with shared dataset loader
        env = _create_session_env(sid)
        obs_dict = _safe_dict(env.reset(seed=seed, difficulty=diff))
        rewards, infos = [], []
        for _ in range(steps):
            if obs_dict.get("done"): break
            ctx = obs_dict.get("context", "")
            action = HallucinationAction(answer=ctx[:100], confidence=0.6, source_quote=ctx[:80])
            obs_dict = _safe_dict(env.step(action))
            rewards.append(float(obs_dict.get("reward") or 0))
            obs_meta = obs_dict.get("metadata", {})
            if isinstance(obs_meta, dict):
                obs_correctness = obs_meta.get("correctness", 0.0)
            else:
                obs_correctness = 0.0
            infos.append({
                "correctness": obs_correctness,
                "grounding": obs_dict.get("grounding_score", 0),
                "calibration": 0.6,
                "hallucination_score": 1.0 if obs_dict.get("is_hallucination") else 0.0,
                "is_hallucination": bool(obs_dict.get("is_hallucination", False)),
            })
        results.append(compute_task_score(task, rewards, infos))
        try: env.close()
        except: pass
    return {"tasks": results, "summary": {"overall_score": round(sum(r["score"] for r in results)/max(len(results),1), 4)}}

@app.post("/batch/evaluate", tags=["Evaluation"])
async def batch_evaluate(body: Dict[str, Any]):
    items = body.get("items", [])
    if not items: raise HTTPException(422, "'items' required")
    from server.grader import calculate_reward
    results = []
    for i, item in enumerate(items):
        r, info = calculate_reward(item.get("answer",""), item.get("confidence",0.5), item.get("source_quote",""), item.get("context",""), item.get("ground_truth",""))
        results.append({"index": i, "reward": round(r,4), "is_hallucination": info.get("is_hallucination", False)})
    return {"total_items": len(results), "results": results}

@app.get("/leaderboard", tags=["Leaderboard"])
async def leaderboard():
    if not _leaderboard: return {"leaderboard": [], "message": "No submissions"}
    ranked = sorted(_leaderboard.values(), key=lambda x: x.get("avg_reward",0), reverse=True)
    for i, e in enumerate(ranked): e["rank"] = i+1
    return {"leaderboard": ranked}

@app.post("/leaderboard/submit", tags=["Leaderboard"])
async def submit_leaderboard(data: Dict[str, Any]):
    required = ["model_name", "avg_reward", "avg_accuracy", "hallucination_rate", "total_episodes", "total_steps"]
    if missing := [f for f in required if f not in data]: raise HTTPException(422, f"Missing: {missing}")
    _leaderboard[data["model_name"]] = {**data, "submitted_at": time.time()}
    _save_leaderboard(_leaderboard)
    return {"status": "submitted", "model_name": data["model_name"]}

@app.get("/health", tags=["Info"])
async def health(): return {"status": "healthy", "version": "4.2.0"}

@app.get("/metadata", tags=["OpenEnv"])
async def metadata():
    return {
        "name": "hallucination-guard-env",
        "version": "4.2.0",
        "license": "MIT",
        "description": (
            "An OpenEnv RL environment that trains AI models to answer questions "
            "ONLY from verified context documents — penalizing hallucination and "
            "rewarding factual grounding."
        ),
    }

@app.get("/schema", tags=["OpenEnv"])
async def schema():
    return {
        "action": {
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer":       {"type": "string",  "description": "Answer derived ONLY from the provided context document."},
                "confidence":   {"type": "number",  "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                "source_quote": {"type": "string",  "default": ""},
                "reasoning":    {"type": "string",  "default": ""},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "question":           {"type": "string"},
                "context":            {"type": "string"},
                "ground_truth":       {"type": "string"},
                "done":               {"type": "boolean"},
                "reward":             {"type": "number"},
                "feedback":           {"type": "string"},
                "is_hallucination":   {"type": "boolean"},
                "grounding_score":    {"type": "number"},
                "difficulty_level":   {"type": "string"},
                "attempts_remaining": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id":            {"type": "string"},
                "step_count":            {"type": "integer"},
                "accuracy":              {"type": "number"},
                "hallucination_rate":    {"type": "number"},
                "average_reward":        {"type": "number"},
                "current_difficulty":    {"type": "string"},
                "skill_rating":          {"type": "number"},
                "current_streak":        {"type": "integer"},
                "best_streak":           {"type": "integer"},
            },
        },
    }

@app.get("/datasets", tags=["Info"])
async def datasets():
    try: return {"total_examples": _get_default_env().dataset_loader.get_total_examples()}
    except: return {"total_examples": 0}

@app.post("/mcp", tags=["OpenEnv"])
async def mcp(body: Dict[str, Any]):
    if body.get("method") == "tools/list":
        return {"jsonrpc": "2.0", "id": body.get("id",1), "result": {"tools": [{"name": "reset", "inputSchema": {"type": "object"}}, {"name": "step", "inputSchema": {"type": "object"}}]}}
    return {"jsonrpc": "2.0", "id": body.get("id",1), "result": {"name": "hallucination-guard-env", "version": "4.2.0"}}

@app.middleware("http")
async def log_req(request, call_next):
    resp = await call_next(request)
    logger.info(f"{request.method} {request.url.path} → {resp.status_code}")
    return resp

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
