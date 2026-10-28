const SAMPLE_RATE = 16000;
const LEAD_S = 0.04; // playback lead to absorb jitter; matches GatewayConfig.client_buffer_ms

const toggle = document.getElementById("toggle");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let ws = null;
let ctx = null;
let micStream = null;
let micNode = null;
let nextTime = 0; // when the next audio frame should start playing
let dropUpTo = 0; // ignore audio for turns <= this (the server told us to stop them)
const playing = new Set();

function setStatus(text) {
  statusEl.textContent = text;
}

function log(text) {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  logEl.prepend(li);
  while (logEl.children.length > 50) logEl.lastChild.remove();
}

async function start() {
  toggle.disabled = true;
  try {
    ctx = new AudioContext(); // created inside the click handler so autoplay rules allow it
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    await ctx.audioWorklet.addModule("mic-worklet.js");
    await ctx.resume();

    micNode = new AudioWorkletNode(ctx, "mic-processor");
    ctx.createMediaStreamSource(micStream).connect(micNode);
    micNode.connect(ctx.destination); // outputs silence; keeps the node in the audio graph

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      micNode.port.onmessage = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
      };
      toggle.textContent = "Stop";
      toggle.disabled = false;
      setStatus("listening");
      log("connected");
    };
    ws.onmessage = onServerMessage;
    ws.onclose = () => {
      log("disconnected");
      stop();
    };
  } catch (err) {
    log(`could not start: ${err.message}`);
    stop();
  }
}

function onServerMessage(ev) {
  if (typeof ev.data === "string") {
    const msg = JSON.parse(ev.data);
    if (msg.type === "stop") {
      stopPlayback();
      dropUpTo = Math.max(dropUpTo, msg.turn_id);
      setStatus("listening");
      log(`agent stopped (turn ${msg.turn_id})`);
    }
    return;
  }
  const turnId = new DataView(ev.data).getUint32(0); // big-endian
  if (turnId <= dropUpTo) return; // late frame from a turn that was already stopped
  playFrame(new Int16Array(ev.data, 4));
}

function playFrame(pcm) {
  const buffer = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);

  if (nextTime < ctx.currentTime) nextTime = ctx.currentTime + LEAD_S; // (re)start with a little lead
  src.start(nextTime);
  nextTime += buffer.duration;

  playing.add(src);
  src.onended = () => {
    playing.delete(src);
    if (playing.size === 0 && ws) setStatus("listening");
  };
  setStatus("agent speaking");
}

function stopPlayback() {
  for (const src of playing) {
    try { src.stop(); } catch (_) { /* already stopped */ }
  }
  playing.clear();
  nextTime = 0;
}

function stop() {
  stopPlayback();
  if (ws) {
    ws.onclose = null;
    ws.close();
    ws = null;
  }
  if (micNode) {
    micNode.port.onmessage = null;
    micNode.disconnect();
    micNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (ctx) {
    ctx.close();
    ctx = null;
  }
  dropUpTo = 0; // turn ids restart with every connection
  toggle.textContent = "Start";
  toggle.disabled = false;
  setStatus("idle");
}

toggle.addEventListener("click", () => (ws ? stop() : start()));