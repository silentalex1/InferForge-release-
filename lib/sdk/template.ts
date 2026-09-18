export const SDK_JS = String.raw`(function (global) {
  "use strict";

  var DEFAULT_ENDPOINT = "{ENDPOINT}";
  var DEFAULT_FALLBACK = "{FALLBACK}";
  var DEFAULT_MODEL = "{MODEL}";
  var DEFAULT_API_KEY = "{API_KEY}";

  function clean(url) {
    return String(url || "").replace(/\/+$/, "");
  }

  function uniq(list) {
    var out = [];
    for (var i = 0; i < list.length; i++) {
      var v = clean(list[i]);
      if (v && out.indexOf(v) < 0) out.push(v);
    }
    return out;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  var InferForge = function (opts) {
    opts = opts || {};
    var origin = (typeof location !== "undefined" && location.origin) || "";
    this.endpoint = clean(opts.endpoint || DEFAULT_ENDPOINT || origin);
    this.fallbacks = uniq(opts.fallbacks || (DEFAULT_FALLBACK ? [DEFAULT_FALLBACK] : []));
    this.model = opts.model || DEFAULT_MODEL;
    this.apiKey = opts.apiKey || DEFAULT_API_KEY || null;
    this.system = opts.system || null;
    this.active = null;
    this.defaults = { temperature: opts.temperature, max_tokens: opts.max_tokens };
  };

  InferForge.prototype.bases = function () {
    if (this.active) return [this.active];
    var out = uniq([this.endpoint].concat(this.fallbacks));
    return out.length ? out : [""];
  };

  InferForge.prototype.headers = function () {
    var h = { "Content-Type": "application/json" };
    if (this.apiKey) h["Authorization"] = "Bearer " + this.apiKey;
    return h;
  };

  InferForge.prototype.payload = function (input, opts, stream) {
    opts = opts || {};
    var msgs = [];
    var system = opts.system || this.system;
    if (system) msgs.push({ role: "system", content: system });
    if (Array.isArray(input)) msgs = msgs.concat(input);
    else msgs.push({ role: "user", content: String(input) });
    var temperature = opts.temperature;
    if (temperature === undefined) temperature = this.defaults.temperature;
    var maxTokens = opts.max_tokens;
    if (maxTokens === undefined) maxTokens = this.defaults.max_tokens;
    return {
      model: opts.model || this.model,
      messages: msgs,
      stream: !!stream,
      temperature: temperature,
      max_tokens: maxTokens
    };
  };

  InferForge.prototype.send = async function (input, opts, stream) {
    var bases = this.bases();
    var last = null;
    for (var i = 0; i < bases.length; i++) {
      var base = bases[i];
      try {
        var res = await fetch(base + "/v1/chat/completions", {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify(this.payload(input, opts, stream))
        });
        if (res.ok) {
          this.active = base;
          return res;
        }
        last = new Error("InferForge " + res.status + ": " + (await res.text()));
      } catch (err) {
        last = err;
      }
    }
    throw last || new Error("InferForge: no reachable endpoint for " + this.model);
  };

  InferForge.prototype.chat = async function (input, opts) {
    var res = await this.send(input, opts || {}, false);
    var data = await res.json();
    var choice = (data.choices && data.choices[0]) || {};
    return (choice.message && choice.message.content) || "";
  };

  InferForge.prototype.chatStream = async function* (input, opts) {
    var res = await this.send(input, opts || {}, true);
    if (!res.body) {
      var data = await res.json();
      var choice = (data.choices && data.choices[0]) || {};
      yield (choice.message && choice.message.content) || "";
      return;
    }
    var reader = res.body.getReader();
    var decoder = new TextDecoder();
    var buf = "";
    while (true) {
      var step = await reader.read();
      if (step.done) break;
      buf += decoder.decode(step.value, { stream: true });
      var lines = buf.split("\n");
      buf = lines.pop() || "";
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        if (line.indexOf("data:") !== 0) continue;
        var raw = line.slice(5).trim();
        if (raw === "[DONE]") return;
        try {
          var chunk = JSON.parse(raw);
          var c = (chunk.choices && chunk.choices[0]) || {};
          var delta = (c.delta && c.delta.content) || "";
          if (delta) yield delta;
        } catch (err) {}
      }
    }
  };

  InferForge.prototype.models = async function () {
    var bases = this.bases();
    for (var i = 0; i < bases.length; i++) {
      try {
        var res = await fetch(bases[i] + "/v1/models", { headers: this.headers() });
        if (!res.ok) continue;
        var data = await res.json();
        return (data.data || []).map(function (m) { return m.id; });
      } catch (err) {}
    }
    return [];
  };

  InferForge.prototype.health = async function () {
    var bases = this.bases();
    for (var i = 0; i < bases.length; i++) {
      try {
        var res = await fetch(bases[i] + "/health");
        if (res.ok) return true;
      } catch (err) {}
    }
    return false;
  };

  InferForge.prototype.mount = function (el, opts) {
    opts = opts || {};
    var root = typeof el === "string" ? document.querySelector(el) : el;
    if (!root) throw new Error("InferForge.mount: element not found");
    var self = this;
    var placeholder = opts.placeholder || "Ask " + this.model;
    root.classList.add("ifw-root");
    root.innerHTML =
      '<div class="ifw-log" style="display:flex;flex-direction:column;gap:8px;overflow-y:auto;flex:1;min-height:0;font:inherit"></div>' +
      '<form class="ifw-form" style="display:flex;gap:8px;margin-top:10px">' +
      '<input class="ifw-input" autocomplete="off" placeholder="' + escapeHtml(placeholder) + '" style="flex:1;min-width:0;padding:9px 12px;border-radius:9px;border:1px solid rgba(127,127,127,.35);background:transparent;color:inherit;font:inherit"/>' +
      '<button class="ifw-send" type="submit" style="padding:9px 16px;border-radius:9px;border:0;cursor:pointer;font:inherit">Send</button>' +
      "</form>";
    if (!root.style.display) root.style.display = "flex";
    if (!root.style.flexDirection) root.style.flexDirection = "column";

    var log = root.querySelector(".ifw-log");
    var form = root.querySelector(".ifw-form");
    var input = root.querySelector(".ifw-input");
    var button = root.querySelector(".ifw-send");

    function push(who, text) {
      var div = document.createElement("div");
      div.className = "ifw-msg ifw-" + who;
      div.style.whiteSpace = "pre-wrap";
      div.style.wordBreak = "break-word";
      div.textContent = (who === "you" ? "You: " : self.model + ": ") + text;
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
      return div;
    }

    var history = this.system ? [{ role: "system", content: this.system }] : [];

    input.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" || e.shiftKey) return;
      e.preventDefault();
      if (typeof form.requestSubmit === "function") form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    });

    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      var q = input.value.trim();
      if (!q || button.disabled) return;
      input.value = "";
      button.disabled = true;
      push("you", q);
      history.push({ role: "user", content: q });
      var bubble = push("ai", "...");
      try {
        var acc = "";
        for await (var chunk of self.chatStream(history, opts)) {
          acc += chunk;
          bubble.textContent = self.model + ": " + acc;
          log.scrollTop = log.scrollHeight;
        }
        if (!acc) acc = await self.chat(history, opts);
        bubble.textContent = self.model + ": " + acc;
        history.push({ role: "assistant", content: acc });
      } catch (err) {
        bubble.textContent = self.model + " is offline: " + (err && err.message ? err.message : String(err));
      } finally {
        button.disabled = false;
        input.focus();
      }
    });

    return root;
  };

  global.InferForge = InferForge;
  if (typeof module !== "undefined" && module.exports) module.exports = InferForge;

  try {
    var tag = (typeof document !== "undefined" && document.currentScript) || null;
    if (tag) {
      var params = null;
      try { params = new URL(tag.src, location.href).searchParams; } catch (err) {}
      var pick = function (name) {
        var fromUrl = params ? params.get(name) : null;
        return fromUrl || tag.getAttribute("data-" + name) || "";
      };
      var key = pick("key");
      var mountSel = pick("mount");
      var system = pick("system");
      var placeholder = pick("placeholder");
      var boot = function () {
        var ai = new InferForge({
          apiKey: key || undefined,
          system: system || undefined
        });
        InferForge.instance = ai;
        if (mountSel) {
          var target = document.querySelector(mountSel);
          if (target) ai.mount(target, { placeholder: placeholder || undefined });
        }
        if (typeof global.onInferForgeReady === "function") global.onInferForgeReady(ai);
        try {
          document.dispatchEvent(new CustomEvent("inferforge:ready", { detail: ai }));
        } catch (err) {}
      };
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
      else boot();
    }
  } catch (err) {}
})(typeof window !== "undefined" ? window : globalThis);
`

export const MISSING_JS = String.raw`(function (global) {
  "use strict";
  var MODEL = "{MODEL}";
  var SITE = "{SITE}";
  var message =
    "InferForge: '" + MODEL + "' is not published yet. " +
    "Run: forge embedd " + MODEL + " --sdk";
  global.InferForge = function () { throw new Error(message); };
  global.InferForge.instance = null;
  if (typeof console !== "undefined") console.error(message, SITE);
  try {
    var tag = document.currentScript;
    var sel = null;
    try { sel = new URL(tag.src, location.href).searchParams.get("mount"); } catch (err) {}
    sel = sel || (tag && tag.getAttribute("data-mount")) || "";
    if (sel) {
      var show = function () {
        var target = document.querySelector(sel);
        if (target) target.textContent = message;
      };
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", show);
      else show();
    }
  } catch (err) {}
})(typeof window !== "undefined" ? window : globalThis);
`
