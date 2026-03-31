"""
HallucinationGuard-Env v4.2 — Production FastAPI Server with Stunning 3D Documentation

Features:
  - Animated 3D particle background
  - Floating geometric objects
  - Glassmorphism UI elements
  - Gradient text and buttons
  - Interactive playground with live testing
  - Smooth animations and transitions

Endpoints:
  Standard   : POST /reset  POST /step  GET /state  GET /health
  Session    : POST /session/reset  POST /session/step  DELETE /session
  Leaderboard: GET /leaderboard  POST /leaderboard/submit
  OpenEnv    : GET /tasks  POST /grader  POST /baseline

"""

import sys, os, uuid, logging, dataclasses, enum, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
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
# STUNNING 3D ANIMATED DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

STUNNING_DOCS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HallucinationGuard-Env | Production RL Environment</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --bg-deep: #030014;
            --bg-primary: #0a0518;
            --bg-secondary: #120826;
            --glass: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #ffffff;
            --text-secondary: rgba(255, 255, 255, 0.7);
            --text-muted: rgba(255, 255, 255, 0.4);
            --accent-1: #7c3aed;
            --accent-2: #06b6d4;
            --accent-3: #f43f5e;
            --accent-4: #10b981;
            --gradient-1: linear-gradient(135deg, #7c3aed 0%, #06b6d4 50%, #10b981 100%);
            --gradient-2: linear-gradient(135deg, #f43f5e 0%, #7c3aed 100%);
            --gradient-3: linear-gradient(135deg, #06b6d4 0%, #10b981 100%);
            --glow-1: 0 0 40px rgba(124, 58, 237, 0.3);
            --glow-2: 0 0 60px rgba(6, 182, 212, 0.2);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            overflow-x: hidden;
            min-height: 100vh;
        }

        /* Three.js Canvas Background */
        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
        }

        /* Animated Gradient Orbs */
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.4;
            animation: float 20s ease-in-out infinite;
            z-index: 1;
            pointer-events: none;
        }

        .orb-1 {
            width: 600px;
            height: 600px;
            background: var(--accent-1);
            top: -200px;
            right: -200px;
            animation-delay: 0s;
        }

        .orb-2 {
            width: 500px;
            height: 500px;
            background: var(--accent-2);
            bottom: -150px;
            left: -150px;
            animation-delay: -5s;
        }

        .orb-3 {
            width: 400px;
            height: 400px;
            background: var(--accent-3);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation-delay: -10s;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(50px, -50px) scale(1.1); }
            50% { transform: translate(-30px, 30px) scale(0.9); }
            75% { transform: translate(-50px, -30px) scale(1.05); }
        }

        /* Grid Pattern Overlay */
        .grid-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: 2;
            pointer-events: none;
        }

        /* Noise Texture */
        .noise {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            opacity: 0.03;
            z-index: 3;
            pointer-events: none;
        }

        /* Main Content Container */
        .content {
            position: relative;
            z-index: 10;
        }

        /* Navigation */
        nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
            padding: 20px 40px;
            background: rgba(3, 0, 20, 0.6);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo-icon {
            width: 44px;
            height: 44px;
            background: var(--gradient-1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: var(--glow-1);
            animation: pulse-glow 3s ease-in-out infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(124, 58, 237, 0.4); }
            50% { box-shadow: 0 0 40px rgba(124, 58, 237, 0.6), 0 0 60px rgba(6, 182, 212, 0.3); }
        }

        .logo-text {
            font-size: 20px;
            font-weight: 600;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .nav-links {
            display: flex;
            gap: 8px;
        }

        .nav-link {
            padding: 10px 20px;
            border-radius: 10px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .nav-link:hover {
            background: var(--glass);
            border-color: var(--glass-border);
            color: var(--text-primary);
        }

        .nav-link.active {
            background: var(--gradient-1);
            color: white;
            box-shadow: var(--glow-1);
        }

        .nav-btn {
            padding: 10px 24px;
            border-radius: 10px;
            background: var(--gradient-2);
            color: white;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: var(--glow-1);
        }

        .nav-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 30px rgba(244, 63, 94, 0.4);
        }

        /* Hero Section */
        .hero {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 120px 40px 80px;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 20px;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 50px;
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 32px;
            backdrop-filter: blur(10px);
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-4);
            border-radius: 50%;
            animation: blink 2s ease-in-out infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; box-shadow: 0 0 10px var(--accent-4); }
            50% { opacity: 0.5; box-shadow: none; }
        }

        .hero h1 {
            font-size: 72px;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 24px;
            background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.7) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInUp 1s ease-out;
        }

        .hero h1 span {
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .hero-subtitle {
            font-size: 22px;
            color: var(--text-secondary);
            max-width: 700px;
            margin-bottom: 48px;
            line-height: 1.6;
            animation: fadeInUp 1s ease-out 0.2s both;
        }

        .hero-buttons {
            display: flex;
            gap: 20px;
            margin-bottom: 80px;
            animation: fadeInUp 1s ease-out 0.4s both;
        }

        .btn {
            padding: 16px 36px;
            border-radius: 14px;
            font-size: 16px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: all 0.3s ease;
            cursor: pointer;
            border: none;
        }

        .btn-primary {
            background: var(--gradient-1);
            color: white;
            box-shadow: var(--glow-1), var(--glow-2);
        }

        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 0 50px rgba(124, 58, 237, 0.5), 0 0 80px rgba(6, 182, 212, 0.3);
        }

        .btn-secondary {
            background: var(--glass);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--accent-1);
            transform: translateY(-3px);
        }

        /* Stats Section */
        .stats-container {
            display: flex;
            justify-content: center;
            gap: 60px;
            flex-wrap: wrap;
            animation: fadeInUp 1s ease-out 0.6s both;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 52px;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }

        .stat-label {
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Floating Elements */
        .floating-shapes {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            pointer-events: none;
            z-index: 5;
            overflow: hidden;
        }

        .shape {
            position: absolute;
            opacity: 0.1;
            animation: shapeFloat 15s ease-in-out infinite;
        }

        .shape-1 { top: 20%; left: 10%; animation-delay: 0s; }
        .shape-2 { top: 60%; left: 80%; animation-delay: -3s; }
        .shape-3 { top: 80%; left: 20%; animation-delay: -6s; }
        .shape-4 { top: 30%; left: 70%; animation-delay: -9s; }
        .shape-5 { top: 70%; left: 50%; animation-delay: -12s; }

        @keyframes shapeFloat {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-30px) rotate(180deg); }
        }

        /* Section Container */
        .section {
            padding: 100px 40px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .section-header {
            text-align: center;
            margin-bottom: 60px;
        }

        .section-title {
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 16px;
            background: linear-gradient(135deg, #fff 0%, rgba(255,255,255,0.8) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .section-subtitle {
            font-size: 18px;
            color: var(--text-secondary);
        }

        /* Glass Cards */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
        }

        .card {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 32px;
            backdrop-filter: blur(20px);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--gradient-1);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .card:hover {
            transform: translateY(-8px);
            border-color: var(--accent-1);
            box-shadow: var(--glow-1), 0 20px 40px rgba(0,0,0,0.3);
        }

        .card:hover::before {
            opacity: 1;
        }

        .card-icon {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin-bottom: 20px;
            position: relative;
        }

        .card-icon.green {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
        }

        .card-icon.yellow {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(249, 115, 22, 0.2) 100%);
            box-shadow: 0 0 30px rgba(251, 191, 36, 0.2);
        }

        .card-icon.red {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(124, 58, 237, 0.2) 100%);
            box-shadow: 0 0 30px rgba(244, 63, 94, 0.2);
        }

        .card-title {
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .card-desc {
            color: var(--text-secondary);
            font-size: 15px;
            line-height: 1.6;
            margin-bottom: 20px;
        }

        .card-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .badge-beginner {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-4);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-intermediate {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }

        .badge-advanced {
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-3);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        /* Playground Section */
        .playground {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            overflow: hidden;
            backdrop-filter: blur(20px);
        }

        .playground-header {
            display: flex;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--glass-border);
        }

        .playground-tab {
            padding: 18px 32px;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }

        .playground-tab:hover {
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.02);
        }

        .playground-tab.active {
            color: var(--accent-1);
            border-bottom-color: var(--accent-1);
            background: rgba(124, 58, 237, 0.05);
        }

        .playground-body {
            display: grid;
            grid-template-columns: 1fr 1fr;
            min-height: 500px;
        }

        .playground-left, .playground-right {
            padding: 32px;
        }

        .playground-left {
            border-right: 1px solid var(--glass-border);
        }

        .playground-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .playground-label::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-1);
            border-radius: 2px;
        }

        .playground-textarea {
            width: 100%;
            height: 280px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 20px;
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            color: var(--text-primary);
            resize: none;
            outline: none;
            transition: all 0.3s ease;
        }

        .playground-textarea:focus {
            border-color: var(--accent-1);
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.2);
        }

        .btn-group {
            display: flex;
            gap: 16px;
            margin-top: 20px;
        }

        .result-box {
            width: 100%;
            height: 380px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 20px;
            font-family: 'Fira Code', monospace;
            font-size: 12px;
            color: var(--text-secondary);
            white-space: pre-wrap;
            overflow-y: auto;
            position: relative;
        }

        .result-box.success {
            border-color: var(--accent-4);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
        }

        .result-box.error {
            border-color: var(--accent-3);
            box-shadow: 0 0 20px rgba(244, 63, 94, 0.1);
        }

        /* Endpoints Table */
        .endpoints-container {
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            overflow: hidden;
            backdrop-filter: blur(20px);
        }

        .endpoint-row {
            display: grid;
            grid-template-columns: 100px 1fr 2fr;
            padding: 20px 32px;
            border-bottom: 1px solid var(--glass-border);
            transition: all 0.3s ease;
            align-items: center;
        }

        .endpoint-row:last-child {
            border-bottom: none;
        }

        .endpoint-row:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .method-badge {
            display: inline-flex;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            font-family: 'Fira Code', monospace;
            letter-spacing: 0.5px;
        }

        .method-get {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-4);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .method-post {
            background: rgba(124, 58, 237, 0.15);
            color: var(--accent-1);
            border: 1px solid rgba(124, 58, 237, 0.3);
        }

        .method-delete {
            background: rgba(244, 63, 94, 0.15);
            color: var(--accent-3);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }

        .endpoint-path {
            font-family: 'Fira Code', monospace;
            font-size: 14px;
            color: var(--text-primary);
            padding-left: 20px;
        }

        .endpoint-desc {
            color: var(--text-secondary);
            font-size: 14px;
            padding-left: 20px;
        }

        /* Features Grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 40px;
        }

        .feature-item {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 24px;
            background: var(--glass);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            transition: all 0.3s ease;
        }

        .feature-item:hover {
            border-color: var(--accent-1);
            transform: translateX(8px);
        }

        .feature-icon {
            width: 40px;
            height: 40px;
            background: var(--gradient-1);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .feature-text h4 {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .feature-text p {
            font-size: 13px;
            color: var(--text-secondary);
        }

        /* Footer */
        footer {
            padding: 60px 40px;
            border-top: 1px solid var(--glass-border);
            text-align: center;
        }

        .footer-text {
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 20px;
        }

        .footer-links {
            display: flex;
            justify-content: center;
            gap: 32px;
            flex-wrap: wrap;
        }

        .footer-link {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 14px;
            transition: color 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .footer-link:hover {
            color: var(--accent-1);
        }

        /* Responsive */
        @media (max-width: 900px) {
            .hero h1 { font-size: 48px; }
            .playground-body { grid-template-columns: 1fr; }
            .playground-left { border-right: none; border-bottom: 1px solid var(--glass-border); }
            .endpoint-row { grid-template-columns: 1fr; gap: 8px; }
            .nav-links { display: none; }
            nav { padding: 16px 20px; }
            .section { padding: 60px 20px; }
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-1); }

        /* Code syntax highlighting */
        .json-key { color: #7c3aed; }
        .json-string { color: #10b981; }
        .json-number { color: #06b6d4; }
    </style>
</head>
<body>
    <!-- Three.js Canvas -->
    <canvas id="bg-canvas"></canvas>

    <!-- Animated Orbs -->
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>

    <!-- Grid Overlay -->
    <div class="grid-overlay"></div>

    <!-- Noise Texture -->
    <div class="noise"></div>

    <!-- Floating Shapes -->
    <div class="floating-shapes">
        <svg class="shape shape-1" width="60" height="60" viewBox="0 0 60 60">
            <polygon points="30,0 60,60 0,60" fill="none" stroke="rgba(124,58,237,0.3)" stroke-width="1"/>
        </svg>
        <svg class="shape shape-2" width="80" height="80" viewBox="0 0 80 80">
            <circle cx="40" cy="40" r="38" fill="none" stroke="rgba(6,182,212,0.3)" stroke-width="1"/>
        </svg>
        <svg class="shape shape-3" width="70" height="70" viewBox="0 0 70 70">
            <rect x="5" y="5" width="60" height="60" fill="none" stroke="rgba(244,63,94,0.3)" stroke-width="1" transform="rotate(45 35 35)"/>
        </svg>
        <svg class="shape shape-4" width="50" height="50" viewBox="0 0 50 50">
            <polygon points="25,0 50,25 25,50 0,25" fill="none" stroke="rgba(16,185,129,0.3)" stroke-width="1"/>
        </svg>
        <svg class="shape shape-5" width="60" height="60" viewBox="0 0 60 60">
            <polygon points="30,0 60,30 30,60 0,30" fill="none" stroke="rgba(124,58,237,0.3)" stroke-width="1"/>
        </svg>
    </div>

    <!-- Content -->
    <div class="content">
        <!-- Navigation -->
        <nav>
            <div class="logo">
                <div class="logo-icon">🛡️</div>
                <span class="logo-text">HallucinationGuard</span>
            </div>
            <div class="nav-links">
                <a href="#overview" class="nav-link">Overview</a>
                <a href="#tasks" class="nav-link">Tasks</a>
                <a href="#playground" class="nav-link active">Playground</a>
                <a href="#endpoints" class="nav-link">Endpoints</a>
            </div>
            <a href="/redoc" class="nav-btn">API Docs →</a>
        </nav>

        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-badge">
                <span class="badge-dot"></span>
                <span>v4.2.0 • OpenEnv Compatible • Production Ready</span>
            </div>
            <h1>Train AI to Stop<br/><span>Hallucinating</span></h1>
            <p class="hero-subtitle">The production-grade RL environment for training and evaluating LLMs on hallucination avoidance. Built on 1M+ real-world examples across 38 benchmark datasets.</p>
            <div class="hero-buttons">
                <a href="#playground" class="btn btn-primary">
                    <span>⚡</span> Try Interactive Demo
                </a>
                <a href="/redoc" class="btn btn-secondary">
                    <span>📖</span> Full API Reference
                </a>
            </div>
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-value" data-count="1090163">0</div>
                    <div class="stat-label">Examples</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" data-count="38">0</div>
                    <div class="stat-label">Datasets</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" data-count="9">0</div>
                    <div class="stat-label">Reward Components</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" data-count="3">0</div>
                    <div class="stat-label">Task Levels</div>
                </div>
            </div>
        </section>

        <!-- Features Section -->
        <section class="section" id="overview">
            <div class="section-header">
                <h2 class="section-title">Why HallucinationGuard?</h2>
                <p class="section-subtitle">Research-grade evaluation for grounded AI systems</p>
            </div>
            <div class="features-grid">
                <div class="feature-item">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-text">
                        <h4>Factual Grounding</h4>
                        <p>Rewards answers derived strictly from provided context</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🔬</div>
                    <div class="feature-text">
                        <h4>9-Component Reward</h4>
                        <p>Factual correctness, grounding, calibration, NLI, BERTScore...</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">📊</div>
                    <div class="feature-text">
                        <h4>Real-World Datasets</h4>
                        <p>SQuAD, HotpotQA, HaluEval, TruthfulQA, FEVER, and 33 more</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-text">
                        <h4>Fast API</h4>
                        <p>RESTful endpoints with OpenEnv compliance</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🧠</div>
                    <div class="feature-text">
                        <h4>NLI-Powered</h4>
                        <p>Detects entailment and contradiction semantically</p>
                    </div>
                </div>
                <div class="feature-item">
                    <div class="feature-icon">🏆</div>
                    <div class="feature-text">
                        <h4>Leaderboard</h4>
                        <p>Compare model performance across tasks</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Tasks Section -->
        <section class="section" id="tasks">
            <div class="section-header">
                <h2 class="section-title">Three Difficulty Levels</h2>
                <p class="section-subtitle">Progressive curriculum from basic to adversarial</p>
            </div>
            <div class="cards-grid">
                <div class="card">
                    <div class="card-icon green">🟢</div>
                    <h3 class="card-title">Task 1: Factual Grounding</h3>
                    <p class="card-desc">Answer straightforward factual questions from a short context passage. Single-hop retrieval with unambiguous ground truth. Perfect for initial training.</p>
                    <span class="card-badge badge-beginner">Beginner</span>
                    <div style="margin-top: 16px; font-size: 12px; color: var(--text-muted);">Datasets: SQuAD, BoolQ, ARC, OpenBookQA</div>
                </div>
                <div class="card">
                    <div class="card-icon yellow">🟡</div>
                    <h3 class="card-title">Task 2: Multi-Hop Synthesis</h3>
                    <p class="card-desc">Synthesize evidence from multiple sentences. Connect disparate facts without fabricating bridging information. Requires reasoning chains.</p>
                    <span class="card-badge badge-intermediate">Intermediate</span>
                    <div style="margin-top: 16px; font-size: 12px; color: var(--text-muted);">Datasets: HotpotQA, CoQA, NQ-Open, MS-MARCO</div>
                </div>
                <div class="card">
                    <div class="card-icon red">🔴</div>
                    <h3 class="card-title">Task 3: Adversarial Resistance</h3>
                    <p class="card-desc">Resist adversarial prompts designed to elicit hallucinations. Many questions are unanswerable — confident refusals are rewarded.</p>
                    <span class="card-badge badge-advanced">Advanced</span>
                    <div style="margin-top: 16px; font-size: 12px; color: var(--text-muted);">Datasets: HaluEval, TruthfulQA, FEVER, AdversarialQA</div>
                </div>
            </div>
        </section>

        <!-- Playground Section -->
        <section class="section" id="playground">
            <div class="section-header">
                <h2 class="section-title">Interactive Playground</h2>
                <p class="section-subtitle">Test the API directly in your browser</p>
            </div>
            <div class="playground">
                <div class="playground-header">
                    <div class="playground-tab active" onclick="switchTab('reset')">🔄 Reset Episode</div>
                    <div class="playground-tab" onclick="switchTab('step')">📝 Submit Answer</div>
                    <div class="playground-tab" onclick="switchTab('batch')">📦 Batch Evaluate</div>
                    <div class="playground-tab" onclick="switchTab('baseline')">🤖 Run Baseline</div>
                </div>
                <div class="playground-body">
                    <div class="playground-left">
                        <div class="playground-label">REQUEST BODY</div>
                        <textarea id="request-body" class="playground-textarea" placeholder="Enter JSON request...">{
  "difficulty": "beginner",
  "seed": 42
}</textarea>
                        <div class="btn-group">
                            <button class="btn btn-primary" onclick="sendRequest()">
                                ▶ Send Request
                            </button>
                            <button class="btn btn-secondary" onclick="clearAll()">
                                Clear
                            </button>
                        </div>
                    </div>
                    <div class="playground-right">
                        <div class="playground-label">RESPONSE</div>
                        <div id="result-box" class="result-box">
<span style="color: var(--text-muted);">// Response will appear here...
//
// Click "Send Request" to test the API</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Endpoints Section -->
        <section class="section" id="endpoints">
            <div class="section-header">
                <h2 class="section-title">All Endpoints</h2>
                <p class="section-subtitle">Complete API reference at a glance</p>
            </div>
            <div class="endpoints-container">
                <div class="endpoint-row">
                    <span class="method-badge method-post">POST</span>
                    <span class="endpoint-path">/reset</span>
                    <span class="endpoint-desc">Start a new episode with optional difficulty and seed</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-post">POST</span>
                    <span class="endpoint-path">/step</span>
                    <span class="endpoint-desc">Submit an answer with confidence and source citation</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-get">GET</span>
                    <span class="endpoint-path">/state</span>
                    <span class="endpoint-desc">Get current episode state, accuracy, and skill rating</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-get">GET</span>
                    <span class="endpoint-path">/tasks</span>
                    <span class="endpoint-desc">List all 3 tasks with complete action schema</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-post">POST</span>
                    <span class="endpoint-path">/grader</span>
                    <span class="endpoint-desc">Score a completed episode (returns 0.0–1.0)</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-post">POST</span>
                    <span class="endpoint-path">/baseline</span>
                    <span class="endpoint-desc">Run built-in heuristic baseline agent</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-post">POST</span>
                    <span class="endpoint-path">/batch/evaluate</span>
                    <span class="endpoint-desc">Evaluate multiple Q&A pairs in one request</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-get">GET</span>
                    <span class="endpoint-path">/leaderboard</span>
                    <span class="endpoint-desc">View ranked model performance</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-get">GET</span>
                    <span class="endpoint-path">/health</span>
                    <span class="endpoint-desc">Service health check</span>
                </div>
                <div class="endpoint-row">
                    <span class="method-badge method-get">GET</span>
                    <span class="endpoint-path">/datasets</span>
                    <span class="endpoint-desc">Dataset statistics and distribution</span>
                </div>
            </div>
        </section>

        <!-- Footer -->
        <footer>
            <p class="footer-text">HallucinationGuard-Env — OpenEnv RL Environment for Hallucination Detection</p>
            <div class="footer-links">
                <a href="https://huggingface.co/spaces/SamSankar/hallucination-guard-env" class="footer-link">🤗 HuggingFace Space</a>
                <a href="https://pypi.org/project/openenv-halluguard/" class="footer-link">📦 PyPI Package</a>
                <a href="/redoc" class="footer-link">📖 API Reference</a>
                <a href="https://github.com/meta-pytorch/OpenEnv" class="footer-link">🔗 OpenEnv</a>
            </div>
        </footer>
    </div>

    <script>
        // ═══════════════════════════════════════════════════════════════════════════════
        // THREE.JS 3D BACKGROUND
        // ═══════════════════════════════════════════════════════════════════════════════

        const canvas = document.getElementById('bg-canvas');
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 30;

        // Particle system
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCount = 2000;
        const posArray = new Float32Array(particlesCount * 3);

        for(let i = 0; i < particlesCount * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 100;
        }

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        const particlesMaterial = new THREE.PointsMaterial({
            size: 0.1,
            color: 0x7c3aed,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });

        const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
        scene.add(particlesMesh);

        // Floating geometric objects
        const geometries = [
            new THREE.IcosahedronGeometry(2, 0),
            new THREE.OctahedronGeometry(2, 0),
            new THREE.TetrahedronGeometry(2, 0),
            new THREE.TorusGeometry(1.5, 0.5, 8, 16),
        ];

        const objects = [];
        const colors = [0x7c3aed, 0x06b6d4, 0xf43f5e, 0x10b981];

        geometries.forEach((geo, i) => {
            const material = new THREE.MeshBasicMaterial({
                color: colors[i],
                wireframe: true,
                transparent: true,
                opacity: 0.3
            });
            const mesh = new THREE.Mesh(geo, material);
            mesh.position.set(
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 20 - 10
            );
            mesh.userData = {
                rotationSpeed: { x: Math.random() * 0.01, y: Math.random() * 0.01 },
                floatSpeed: Math.random() * 0.02 + 0.01,
                floatOffset: Math.random() * Math.PI * 2
            };
            objects.push(mesh);
            scene.add(mesh);
        });

        // Mouse movement effect
        let mouseX = 0, mouseY = 0;
        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX / window.innerWidth) * 2 - 1;
            mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
        });

        // Animation loop
        let time = 0;
        function animate() {
            requestAnimationFrame(animate);
            time += 0.01;

            particlesMesh.rotation.y += 0.001;
            particlesMesh.rotation.x += 0.0005;

            // Camera follows mouse slightly
            camera.position.x += (mouseX * 3 - camera.position.x) * 0.02;
            camera.position.y += (mouseY * 3 - camera.position.y) * 0.02;
            camera.lookAt(scene.position);

            // Animate floating objects
            objects.forEach((obj, i) => {
                obj.rotation.x += obj.userData.rotationSpeed.x;
                obj.rotation.y += obj.userData.rotationSpeed.y;
                obj.position.y += Math.sin(time + obj.userData.floatOffset) * 0.02;
            });

            renderer.render(scene, camera);
        }
        animate();

        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // PLAYGROUND FUNCTIONALITY
        // ═══════════════════════════════════════════════════════════════════════════════

        let currentTab = 'reset';
        const endpoints = {
            reset: '/reset',
            step: '/step',
            batch: '/batch/evaluate',
            baseline: '/baseline'
        };

        const placeholders = {
            reset: `{
  "difficulty": "beginner",
  "seed": 42
}`,
            step: `{
  "answer": "Your answer derived from context",
  "confidence": 0.85,
  "source_quote": "Exact quote from context"
}`,
            batch: `{
  "items": [
    {
      "question": "What is the capital of France?",
      "context": "The capital of France is Paris.",
      "answer": "Paris",
      "confidence": 0.9,
      "ground_truth": "Paris"
    }
  ],
  "task_id": "task_1_factual_grounding"
}`,
            baseline: `{
  "steps_per_task": 5,
  "seed": 42
}`
        };

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.playground-tab').forEach(t => {
                t.classList.toggle('active', t.textContent.toLowerCase().includes(tab));
            });
            document.getElementById('request-body').value = placeholders[tab];
            document.getElementById('result-box').innerHTML = '<span style="color: var(--text-muted);">// Response will appear here...</span>';
            document.getElementById('result-box').className = 'result-box';
        }

        async function sendRequest() {
            const body = document.getElementById('request-body').value;
            const resultBox = document.getElementById('result-box');

            try {
                resultBox.innerHTML = '<span style="color: var(--accent-2);">⏳ Sending request...</span>';

                const response = await fetch(endpoints[currentTab], {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: body
                });

                const data = await response.json();
                resultBox.className = 'result-box success';
                resultBox.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                resultBox.className = 'result-box error';
                resultBox.textContent = 'Error: ' + error.message;
            }
        }

        function clearAll() {
            document.getElementById('request-body').value = placeholders[currentTab];
            document.getElementById('result-box').innerHTML = '<span style="color: var(--text-muted);">// Response will appear here...</span>';
            document.getElementById('result-box').className = 'result-box';
        }

        // ════════════════════���══════════════════════════════════════════════════════════
        // ANIMATED COUNTERS
        // ═══════════════════════════════════════════════════════════════════════════════

        function animateCounters() {
            const counters = document.querySelectorAll('.stat-value[data-count]');
            counters.forEach(counter => {
                const target = parseInt(counter.getAttribute('data-count'));
                const duration = 2000;
                const start = performance.now();

                function update(currentTime) {
                    const elapsed = currentTime - start;
                    const progress = Math.min(elapsed / duration, 1);
                    const easeOut = 1 - Math.pow(1 - progress, 3);
                    const current = Math.floor(easeOut * target);

                    counter.textContent = current.toLocaleString();

                    if (progress < 1) {
                        requestAnimationFrame(update);
                    } else {
                        counter.textContent = target >= 1000000 ? '1M+' : target.toLocaleString();
                    }
                }

                requestAnimationFrame(update);
            });
        }

        // Intersection Observer for counter animation
        const statsObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCounters();
                    statsObserver.disconnect();
                }
            });
        }, { threshold: 0.5 });

        const statsContainer = document.querySelector('.stats-container');
        if (statsContainer) {
            statsObserver.observe(statsContainer);
        }

        // Smooth scroll for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    </script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

_default_env: Optional[HallucinationEnvironment] = None
_env_loading = False
_env_lock = threading.Lock()

def _get_default_env() -> HallucinationEnvironment:
    global _default_env, _env_loading
    if _default_env is not None:
        return _default_env
    with _env_lock:
        if _default_env is not None:
            return _default_env
        _env_loading = True
        try:
            logger.info("Creating HallucinationEnvironment...")
            _default_env = HallucinationEnvironment()
            logger.info(f"Environment ready — {_default_env.dataset_loader.get_total_examples():,} examples loaded.")
            return _default_env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            # Minimal fallback environment
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _default_env

    def preload_models():
        try:
            logger.info("Preloading ML models...")
            from sentence_transformers import SentenceTransformer, CrossEncoder
            SentenceTransformer('all-MiniLM-L6-v2')
            CrossEncoder('cross-encoder/nli-deberta-v3-small')
            from rouge_score import rouge_scorer
            rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            try:
                from bert_score import BERTScorer
                BERTScorer(model_type='microsoft/deberta-v3-base', lang='en', device='cpu')
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

_sessions: Dict[str, HallucinationEnvironment] = {}
import json as _json
_LEADERBOARD_FILE = "/tmp/hallucination_guard_leaderboard.json"

def _load_leaderboard():
    if os.path.exists(_LEADERBOARD_FILE):
        try: return _json.load(open(_LEADERBOARD_FILE))
        except: pass
    return {}

def _save_leaderboard(lb):
    try: _json.dump(lb, open(_LEADERBOARD_FILE, "w"), indent=2)
    except: pass

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

@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def docs(): return STUNNING_DOCS_HTML

@app.post("/reset", tags=["Environment"])
async def reset(body: Dict[str, Any] = {}):
    try:
        env = _get_default_env()
        obs = env.reset(**{k: v for k, v in body.items() if k in ("seed", "episode_id", "difficulty")})
        return JSONResponse(content=_safe_dict(obs))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/step", tags=["Environment"])
async def step(action_data: Dict[str, Any]):
    try:
        env = _get_default_env()
        valid = set(HallucinationAction.model_fields.keys()) if hasattr(HallucinationAction, 'model_fields') else set(HallucinationAction.__fields__.keys())
        action = HallucinationAction(**{k: v for k, v in action_data.items() if k in valid})
        return JSONResponse(content=_safe_dict(env.step(action)))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/state", tags=["Environment"])
async def get_state():
    try:
        return JSONResponse(content=_safe_dict(_get_default_env().state()))
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
        if sid in _sessions: _sessions[sid].close()
        _sessions[sid] = HallucinationEnvironment(session_id=sid)
        obs = _safe_dict(_sessions[sid].reset(seed=seed, difficulty=diff))
        rewards, infos = [], []
        for _ in range(steps):
            if obs.get("done"): break
            ctx = obs.get("context", "")
            action = HallucinationAction(answer=ctx[:100], confidence=0.6, source_quote=ctx[:80])
            obs = _safe_dict(_sessions[sid].step(action))
            rewards.append(float(obs.get("reward") or 0))
            infos.append({"correctness": obs.get("grounding_score", 0), "is_hallucination": obs.get("is_hallucination", False)})
        results.append(compute_task_score(task, rewards, infos))
        try: _sessions[sid].close(); del _sessions[sid]
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
async def metadata(): return {"name": "hallucination-guard-env", "version": "4.2.0", "license": "MIT"}

@app.get("/schema", tags=["OpenEnv"])
async def schema(): return {"action": {"type": "object", "required": ["answer"]}, "observation": {"type": "object"}}

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
