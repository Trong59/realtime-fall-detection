const el = (id) => document.getElementById(id);

let statusPollTimer = null;

// ---------------------------------------------------------------- Sliders
function bindSlider(inputId, labelId, decimals) {
  const input = el(inputId);
  const label = el(labelId);
  input.addEventListener("input", () => {
    label.textContent = decimals > 0 ? Number(input.value).toFixed(decimals) : input.value;
  });
}
bindSlider("threshold", "thresholdVal", 2);
bindSlider("consecutive", "consecutiveVal", 0);
bindSlider("cooldown", "cooldownVal", 1);
bindSlider("staticDuration", "staticVal", 1);

// ------------------------------------------------------------ Source mode
document.querySelectorAll('input[name="sourceMode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const isFile = document.querySelector('input[name="sourceMode"]:checked').value === "file";
    el("videoSelect").disabled = !isFile;
  });
});

async function loadVideoList() {
  try {
    const res = await fetch("/api/videos");
    const data = await res.json();
    const select = el("videoSelect");
    select.innerHTML = "";
    if (data.videos.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "(không tìm thấy video trong thư mục test)";
      select.appendChild(opt);
      return;
    }
    data.videos.forEach((path) => {
      const opt = document.createElement("option");
      opt.value = path;
      opt.textContent = path;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error("Không tải được danh sách video:", e);
  }
}

// --------------------------------------------------------------- Start/Stop
el("startBtn").addEventListener("click", async () => {
  const sourceMode = document.querySelector('input[name="sourceMode"]:checked').value;
  const payload = {
    source_mode: sourceMode,
    file_path: sourceMode === "file" ? el("videoSelect").value : null,
    threshold: parseFloat(el("threshold").value),
    consecutive_frames: parseInt(el("consecutive").value, 10),
    cooldown: parseFloat(el("cooldown").value),
    static_duration: parseFloat(el("staticDuration").value),
    enable_safety_net: el("enableSafetyNet").checked,
    enable_sound: el("enableSound").checked,
  };

  const res = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (res.ok) {
    el("startBtn").disabled = true;
    el("stopBtn").disabled = false;
    // Nạp lại luồng MJPEG (cache-bust) để chắc chắn nối vào server đang chạy
    el("videoFeed").src = "/video_feed?t=" + Date.now();
    startStatusPolling();
  } else {
    alert("Lỗi: " + data.message);
  }
});

el("stopBtn").addEventListener("click", async () => {
  await fetch("/api/stop", { method: "POST" });
  el("startBtn").disabled = false;
  el("stopBtn").disabled = true;
  stopStatusPolling();
  setStatusBanner("stopped", "TRẠNG THÁI: ĐÃ DỪNG");
});

// ----------------------------------------------------------- Status polling
function startStatusPolling() {
  if (statusPollTimer) return;
  statusPollTimer = setInterval(pollStatus, 300);
}

function stopStatusPolling() {
  clearInterval(statusPollTimer);
  statusPollTimer = null;
}

function setStatusBanner(kind, text) {
  const banner = el("statusBanner");
  banner.className = "status-banner status-" + kind;
  banner.textContent = text;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    el("fpsVal").textContent = data.fps ? data.fps.toFixed(1) : "-";
    el("probVal").textContent = data.probability !== null && data.probability !== undefined
      ? data.probability.toFixed(2) : "N/A";
    el("staticDurVal").textContent = (data.static_duration || 0).toFixed(1) + "s";

    if (data.error) {
      alert("Lỗi pipeline: " + data.error);
      stopStatusPolling();
      el("startBtn").disabled = false;
      el("stopBtn").disabled = true;
      return;
    }

    if (!data.running) {
      setStatusBanner("stopped", "TRẠNG THÁI: ĐÃ DỪNG (hết video?)");
      el("startBtn").disabled = false;
      el("stopBtn").disabled = true;
      stopStatusPolling();
    } else if (data.lstm_alert_active) {
      setStatusBanner("alert-lstm", "⚠ CẢNH BÁO TÉ NGÃ!");
    } else if (data.static_alert_active) {
      setStatusBanner("alert-static", "⚠ NGHI NGỜ BẤT ĐỘNG - KIỂM TRA NGAY");
    } else {
      setStatusBanner("normal", "TRẠNG THÁI: BÌNH THƯỜNG");
    }

    el("alertLog").textContent = data.alerts.join("\n");
  } catch (e) {
    console.error("Lỗi polling status:", e);
  }
}

// --------------------------------------------------------------------- Init
loadVideoList();
