# BBMP Civic Complaint System — Frontend Design System

## 🎨 Visual Identity & Aesthetic
The goal is to create a **premium, state-of-the-art** interface that feels trustworthy, modern, and high-tech. Avoid generic government-style designs.

### Color Palette (Nature & Technology)
- **Primary (Electric Indigo):** `hsl(245, 80%, 65%)`
- **Secondary (Vibrant Cyan):** `hsl(180, 75%, 50%)`
- **Accent (Gold Leaf):** `hsl(45, 90%, 60%)`
- **Background (Deep Space):** `hsl(220, 30%, 5%)`
- **Surface (Glassmorphism):** `hsla(220, 20%, 15%, 0.7)` with `backdrop-filter: blur(12px)`
- **Success (Emerald):** `hsl(150, 70%, 55%)`
- **Error (Crimson):** `hsl(0, 75%, 60%)`

### Typography (Modern & Legible)
- **Headings:** 'Outfit', sans-serif (700 weight for H1, 600 for H2)
- **Body:** 'Inter', sans-serif (400 for regular, 500 for medium)
- **Monospace:** 'JetBrains Mono' (for status IDs/Coordinates)

---

## ⚡ UI Components & Interactions

### 1. The "Glass" Navbar
- **Effect:** Sticky header with `backdrop-filter: blur(16px)` and a subtle 1px border bottom (`rgba(255,255,255,0.1)`).
- **Navigation:** Hover effects using "Underline slide" animation starting from the center.

### 2. Multi-Lingual Hero Section (Citizen Portal)
- **Background:** A slow-moving mesh gradient of Indigo and Dark Blue.
- **CTA:** A central "Voice Capture" button that pulsates with a glow when active. Includes a real-time waveform visualization using Canvas/SVG.

### 3. Complaint Cards (Admin Dashboard)
- **Shape:** Rounded corners (`1.5rem`).
- **Shadows:** Soft deep-blue glow on hover (`0 20px 40px rgba(0,0,0,0.4)`).
- **Status Pills:** Glowing neon borders matching the status (Pending = Amber, Verified = Emerald).
- **Interactions:** Subtle scale-up (`scale(1.02)`) on hover.

---

## 🗺️ Interactive Maps (Leaflet Customization)
- **Theme:** Use a "Dark Matter" tile layer (CartoDB Dark).
- **Markers:** Pulsating rings for "Pending" complaints and sharp, glowing pins for "Verified" complaints.
- **Clusters:** Use `MarkerCluster` with a custom SVG donut chart showing category distribution in that area.

---

## 📱 Micro-Animations
- **Page Transitions:** Slide-up and fade-in using `framer-motion` or CSS keyframes.
- **Button Feedback:** "Liquid" ripple effect on click.
- **Success State:** Confetti burst or a "checkmark draw" SVG animation when a complaint is verified.

---

## 📜 SEO & Accessibility
- **H1:** "Voice-Enabled Civic Complaint System for Bengaluru"
- **Meta:** "A multilingual, AI-powered platform for reporting civic issues in Bengaluru (BBMP)."
- **Contrast:** AA/AAA compliant text colors against the dark background.
