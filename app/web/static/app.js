"use strict";

const POLL_MS = 500;
const $ = (id) => document.getElementById(id);

const ui = {
  video: $("video"),
  placeholder: $("placeholder"),
  placeholderTitle: $("placeholder-title"),
  placeholderText: $("placeholder-text"),
  notice: $("notice"),
  flash: $("flash"),
  start: $("start"),
  stop: $("stop"),
  camera: $("camera"),
  findCameras: $("find-cameras"),
  scanFace: $("scan-face"),
  registerOpen: $("register-open"),
  registerForm: $("register-form"),
  registerCancel: $("register-cancel"),
  profileName: $("profile-name"),
  idActions: $("id-actions"),
  idProgress: $("id-progress"),
  idBarFill: $("id-bar-fill"),
  idProgressText: $("id-progress-text"),
  idCancel: $("id-cancel"),
  idResult: $("id-result"),
  idBanner: $("id-banner"),
  profiles: $("profiles"),
  reset: $("reset"),
  capture: $("capture"),
  filters: $("filters"),
  landmarks: $("landmarks"),
  axes: $("axes"),
  cameraPill: $("camera-pill"),
  facePill: $("face-pill"),
  fps: $("fps"),
  latency: $("latency"),
  yaw: $("yaw"),
  pitch: $("pitch"),
  roll: $("roll"),
  toast: $("toast"),
  toastImage: $("toast-image"),
  toastText: $("toast-text"),
};

let current = null;
let streaming = false;
let busy = false;
let filtersBuilt = false;
let lastError = null;
let toastTimer = null;
let toastUrl = null;
let registering = false;
let shownResultId = null;
let bannerTimer = null;
let profilesSignature = null;

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return response;
}

function setPill(pill, tone, text) {
  pill.className = `pill ${tone}`.trim();
  pill.querySelector("span").textContent = text;
}

function buildFilters(filters) {
  filters.forEach((filter, position) => {
    const card = document.createElement("button");
    card.className = "filter-card";
    card.dataset.key = filter.key;
    card.disabled = !filter.available;
    card.title = filter.available ? `${filter.label} (press ${position + 1})` : filter.problem;

    const image = document.createElement("img");
    image.src = `/assets/filters/thumb_${filter.key}.png`;
    image.alt = "";
    image.addEventListener("error", () => image.remove());

    const label = document.createElement("span");
    label.textContent = filter.available ? filter.label : `${filter.label} (unavailable)`;

    card.append(image, label);
    card.addEventListener("click", () => selectFilter(filter.key));
    ui.filters.append(card);
  });
  filtersBuilt = true;
}

function renderCameras(state) {
  if (state.demo) {
    ui.camera.replaceChildren(new Option("Demo image", "0"));
    ui.camera.disabled = true;
    return;
  }
  const signature = state.cameras.join(",");
  if (ui.camera.dataset.signature !== signature) {
    ui.camera.replaceChildren(...state.cameras.map((i) => new Option(`Camera ${i}`, String(i))));
    ui.camera.dataset.signature = signature;
    ui.camera.value = String(state.camera);
  }
}

function setStreaming(on) {
  if (on === streaming) return;
  streaming = on;
  if (on) {
    ui.video.src = `/stream?t=${Date.now()}`;
  } else {
    // Removing the source closes the connection to the video stream.
    ui.video.removeAttribute("src");
  }
}

function faceStatus(state) {
  if (state.state !== "running") return ["", "No face"];
  if (state.faces > 1) return ["warn", "2 faces: using largest"];
  if (state.tracking) return ["ok", "Face detected"];
  return ["warn", "No face in view"];
}

function resultTone(result) {
  return result.kind === "recognized" || result.kind === "enrolled" ? "good" : "bad";
}

function resultText(result) {
  if (result.score === undefined) return result.message;
  return `${result.message} Match score ${result.score.toFixed(2)} (needs ${result.threshold.toFixed(2)}).`;
}

function renderProfiles(profiles) {
  const signature = JSON.stringify(profiles);
  if (signature === profilesSignature) return;
  profilesSignature = signature;
  ui.profiles.replaceChildren(...profiles.map((name) => {
    const row = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = name;
    label.title = name;
    const remove = document.createElement("button");
    remove.className = "link-button";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete the face profile for ${name}`);
    remove.addEventListener("click", () => deleteProfile(name));
    row.append(label, remove);
    return row;
  }));
}

function showBanner(text, tone) {
  ui.idBanner.textContent = text;
  ui.idBanner.className = `id-banner ${tone}`.trim();
  ui.idBanner.hidden = false;
}

function renderIdentity(identity) {
  const { session, result, problem } = identity;
  renderProfiles(identity.profiles);

  ui.idProgress.hidden = !session;
  ui.idActions.hidden = Boolean(session) || registering;
  ui.registerForm.hidden = Boolean(session) || !registering;

  if (session) {
    clearTimeout(bannerTimer);
    bannerTimer = null;
    const percent = Math.round((session.collected / session.total) * 100);
    ui.idBarFill.style.width = `${percent}%`;
    const verb = session.mode === "enroll" ? "Registering" : "Scanning";
    ui.idProgressText.textContent = `${verb}: ${session.collected} of ${session.total}`;
    showBanner(session.hint, "");
  } else if (!bannerTimer) {
    ui.idBanner.hidden = true;
  }

  ui.idResult.hidden = !(result || problem);
  if (result) {
    ui.idResult.textContent = resultText(result);
    ui.idResult.className = `id-result ${resultTone(result)}`;
    if (result.id !== shownResultId) {
      shownResultId = result.id;
      showBanner(result.message, resultTone(result));
      clearTimeout(bannerTimer);
      bannerTimer = setTimeout(() => {
        bannerTimer = null;
        ui.idBanner.hidden = true;
      }, 6000);
    }
  } else if (problem) {
    ui.idResult.textContent = problem;
    ui.idResult.className = "id-result bad";
  }
}

function render(state) {
  current = state;
  if (!filtersBuilt) buildFilters(state.filters);
  renderCameras(state);
  renderIdentity(state.identity);

  const running = state.state === "running";
  const starting = state.state === "starting" || busy;
  setStreaming(running);

  ui.start.hidden = running;
  ui.stop.hidden = !running;
  ui.start.disabled = starting;
  ui.start.textContent = starting ? "Starting…" : "Start Camera";
  ui.capture.disabled = !running;
  if (!state.demo) ui.camera.disabled = running || starting;

  if (running) {
    setPill(ui.cameraPill, "ok", state.source || "Camera on");
  } else if (starting) {
    setPill(ui.cameraPill, "warn", "Starting");
  } else if (state.state === "error" || lastError) {
    setPill(ui.cameraPill, "bad", "Camera error");
  } else {
    setPill(ui.cameraPill, "", "Camera off");
  }
  const [faceTone, faceText] = faceStatus(state);
  setPill(ui.facePill, faceTone, faceText);

  ui.fps.textContent = running && state.fps ? state.fps.toFixed(0) : "–";
  ui.latency.textContent = running && state.process_ms ? state.process_ms.toFixed(0) : "–";

  ui.placeholder.hidden = running;
  const message = lastError || state.message;
  ui.placeholder.classList.toggle("error", Boolean(lastError) || state.state === "error");
  if (starting) {
    ui.placeholderTitle.textContent = "Starting camera…";
    ui.placeholderText.textContent = "This can take a few seconds.";
  } else if (lastError || state.state === "error") {
    ui.placeholderTitle.textContent = "Camera problem";
    ui.placeholderText.textContent = message;
  } else {
    ui.placeholderTitle.textContent = "Camera is off";
    ui.placeholderText.textContent = message || "Press Start Camera to begin. Video stays on this computer.";
  }

  ui.notice.hidden = !(running && !state.tracking);
  ui.notice.textContent = "No face in view. Move into the frame to see filters.";

  for (const card of ui.filters.children) {
    card.setAttribute("aria-pressed", String(card.dataset.key === state.filter));
  }
  ui.landmarks.checked = state.landmarks;
  ui.axes.checked = state.axes;

  const format = (v) => (state.pose ? `${v.toFixed(0)}°` : "–");
  ui.yaw.textContent = format(state.pose?.yaw);
  ui.pitch.textContent = format(state.pose?.pitch);
  ui.roll.textContent = format(state.pose?.roll);
}

function renderOffline() {
  setStreaming(false);
  ui.placeholder.hidden = false;
  ui.placeholder.classList.add("error");
  ui.placeholderTitle.textContent = "Lost connection to the app";
  ui.placeholderText.textContent = "The local server isn't running. Start it again, then reload this page.";
  setPill(ui.cameraPill, "bad", "Server offline");
  setPill(ui.facePill, "", "No face");
  ui.start.hidden = false;
  ui.stop.hidden = true;
  ui.start.disabled = true;
  ui.capture.disabled = true;
  ui.fps.textContent = "–";
  ui.latency.textContent = "–";
}

async function refresh() {
  try {
    const response = await fetch("/api/state");
    render(await response.json());
  } catch {
    renderOffline();
  }
}

async function run(action) {
  try {
    await action();
  } catch (error) {
    showToast(error.message);
  }
  await refresh();
}

async function startCamera() {
  busy = true;
  lastError = null;
  if (current) render(current);
  try {
    await post("/api/start", { camera: Number(ui.camera.value || 0) });
  } catch (error) {
    lastError = error.message;
  }
  busy = false;
  await refresh();
}

function selectFilter(key) {
  run(() => post("/api/filter", { key }));
}

async function findCameras() {
  ui.findCameras.disabled = true;
  ui.findCameras.textContent = "Searching…";
  try {
    const response = await post("/api/cameras/find");
    const { cameras } = await response.json();
    showToast(cameras.length ? `Found ${cameras.length} camera${cameras.length > 1 ? "s" : ""}.` : "No working cameras found.");
  } catch (error) {
    showToast(error.message);
  }
  ui.findCameras.disabled = false;
  ui.findCameras.textContent = "Find Cameras";
  await refresh();
}

async function openRegisterForm() {
  // The page's status can be up to one poll old, so check again before deciding.
  await refresh();
  if (current?.state !== "running") {
    showToast("Start the camera first.");
    return;
  }
  registering = true;
  if (current) renderIdentity(current.identity);
  ui.profileName.focus();
}

function closeRegisterForm() {
  registering = false;
  ui.profileName.value = "";
  if (current) renderIdentity(current.identity);
}

async function startRegistration(event) {
  event.preventDefault();
  const name = ui.profileName.value;
  try {
    await post("/api/identity/enroll", { name });
    closeRegisterForm();
  } catch (error) {
    showToast(error.message);
  }
  await refresh();
}

function deleteProfile(name) {
  if (!window.confirm(`Delete the saved face template for ${name}?`)) return;
  run(async () => {
    await post("/api/identity/delete", { name });
    showToast(`Deleted the face template for ${name}.`);
  });
}

function timestamp() {
  const now = new Date();
  const pad = (v) => String(v).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

async function capturePhoto() {
  if (ui.capture.disabled) return;
  try {
    const response = await post("/api/capture");
    const blob = await response.blob();

    // The browser saves the file, so the server never writes photos to disk.
    if (toastUrl) URL.revokeObjectURL(toastUrl);
    toastUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = toastUrl;
    link.download = `ar-filter-${timestamp()}.png`;
    link.click();

    ui.flash.classList.remove("on");
    void ui.flash.offsetWidth;
    ui.flash.classList.add("on");
    showToast("Photo sent to your browser's Downloads.", toastUrl);
  } catch (error) {
    showToast(error.message);
  }
}

function showToast(text, imageUrl) {
  ui.toastText.textContent = text;
  if (imageUrl) ui.toastImage.src = imageUrl;
  else ui.toastImage.removeAttribute("src");
  ui.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { ui.toast.hidden = true; }, 3500);
}

ui.start.addEventListener("click", startCamera);
ui.stop.addEventListener("click", () => run(() => post("/api/stop")));
ui.reset.addEventListener("click", () => run(() => post("/api/reset")));
ui.findCameras.addEventListener("click", findCameras);
ui.scanFace.addEventListener("click", () => run(() => post("/api/identity/scan")));
ui.registerOpen.addEventListener("click", openRegisterForm);
ui.registerCancel.addEventListener("click", closeRegisterForm);
ui.registerForm.addEventListener("submit", startRegistration);
ui.idCancel.addEventListener("click", () => run(() => post("/api/identity/cancel")));
ui.capture.addEventListener("click", capturePhoto);
ui.landmarks.addEventListener("change", () => run(() => post("/api/options", { landmarks: ui.landmarks.checked })));
ui.axes.addEventListener("change", () => run(() => post("/api/options", { axes: ui.axes.checked })));

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.target.matches("select, input")) return;
  if (event.key.toLowerCase() === "c") {
    capturePhoto();
    return;
  }
  const card = ui.filters.children[Number(event.key) - 1];
  if (card && !card.disabled) selectFilter(card.dataset.key);
});

refresh();
setInterval(refresh, POLL_MS);
