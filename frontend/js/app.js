
const $ = (s) => document.querySelector(s);
let token = localStorage.getItem("ov_token");
let user = JSON.parse(localStorage.getItem("ov_user") || "null");
let selectedFile = null;
let latestReportHtml = "";

function showAuth(){
  $("#auth").hidden = false;
  $("#app").hidden = true;
}
function showApp(){
  $("#auth").hidden = true;
  $("#app").hidden = false;
  const name = (user?.name || "").trim();
  $("#userName") && ($("#userName").textContent = name);
  $("#sideUser").textContent = name || "Screening account";
  $("#welcomeName").textContent = name || "Screening user";
  $("#topUser").textContent = name || "Screening user";
  $("#avatar").textContent = name ? name.charAt(0).toUpperCase() : "S";
  if($("#topAvatar")) $("#topAvatar").textContent = name ? name.charAt(0).toUpperCase() : "S";
  syncProfile();
  navigate("dashboard");
  loadHistory();
}
function showAuthError(message){
  $("#authError").textContent = message;
  $("#authError").hidden = !message;
}
function navigate(page){
  document.querySelectorAll(".page").forEach(p => p.hidden = true);
  const target=$("#" + page);
  if(!target) return;
  target.hidden = false;
  const contexts={dashboard:["Dashboard","Retinal screening workspace"],screening:["New screening","AI-assisted retinal image assessment"],history:["Screening history","Saved screening activity and review details"],reports:["Reports","Screening reports and clinical review documents"],profile:["Profile","Account and workspace settings"],help:["Help & support","Assistance for screening, results and reports"]};
  const meta=contexts[page] || ["OcuVisionAI","Retinal screening workspace"];
  $("#pageContext").textContent = meta[0];
  if($("#pageSubtitle")) $("#pageSubtitle").textContent = meta[1];
  document.title = `OcuVisionAI | ${meta[0]}`;
  document.querySelectorAll(".nav").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  if(page === "profile") syncProfile();
  if(page === "history") loadHistory();
  if(page === "reports") loadReports();
  if(page === "dashboard") loadHistory();
  window.scrollTo({top:0, behavior:"smooth"});
}

const sidebarCollapsed = localStorage.getItem("ov_sidebar_collapsed") === "1";
function setSidebarCollapsed(collapsed){
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  const b = $("#sidebarToggle");
  if(b){ b.innerHTML = collapsed ? '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3"></rect><path d="M15 4v16M11 9l3 3-3 3"></path></svg>' : '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="4" y="4" width="16" height="16" rx="3"></rect><path d="M9 4v16M13 9l-3 3 3 3"></path></svg>'; b.title = collapsed ? "Expand sidebar" : "Collapse sidebar"; b.setAttribute("aria-label", b.title); }
  localStorage.setItem("ov_sidebar_collapsed", collapsed ? "1" : "0");
}
setSidebarCollapsed(sidebarCollapsed);
if($("#sidebarToggle")) $("#sidebarToggle").onclick = () => setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));

function showLoginForm(){
  $("#loginForm").hidden = false;
  $("#registerForm").hidden = true;
  $("#loginMeta").hidden = false;
  $("#loginCreateRow").hidden = false;
  $("#authTitle").textContent = "Sign in";
  showAuthError("");
}
function showRegisterForm(){
  $("#loginForm").hidden = true;
  $("#registerForm").hidden = false;
  $("#loginMeta").hidden = true;
  $("#loginCreateRow").hidden = true;
  $("#authTitle").textContent = "Create account";
  showAuthError("");
}
$("#loginTab").onclick = showLoginForm;
$("#registerTab").onclick = showRegisterForm;
$("#forgotPassword").onclick = () => {
  showAuthError("Password reset is not configured in this local workspace yet. Please use your registered password or contact the workspace administrator.");
  $("#authError").className = "form-error";
};

$("#loginForm").onsubmit = async (e) => {
  e.preventDefault();
  showAuthError("");
  const button = e.submitter;
  button.disabled = true;
  button.textContent = "Signing in...";
  try{
    const r = await fetch("/api/login", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        email:$("#loginEmail").value.trim(),
        password:$("#loginPassword").value
      })
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || "Login failed.");
    token = d.token;
    user = d.user;
    localStorage.setItem("ov_token", token);
    localStorage.setItem("ov_user", JSON.stringify(user));
    showApp();
  }catch(err){
    showAuthError(err.message);
  }finally{
    button.disabled = false;
    button.textContent = "Sign in";
  }
};

$("#registerForm").onsubmit = async (e) => {
  e.preventDefault();
  showAuthError("");
  const button = e.submitter;
  button.disabled = true;
  button.textContent = "Creating account...";
  try{
    const r = await fetch("/api/register", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        name:$("#regName").value.trim(),
        email:$("#regEmail").value.trim(),
        password:$("#regPassword").value
      })
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || "Registration failed.");
    $("#loginTab").click();
    $("#loginEmail").value = $("#regEmail").value;
    $("#loginPassword").value = "";
    showAuthError("Account created. Please sign in.");
    $("#authError").className = "form-error success-message";
  }catch(err){
    showAuthError(err.message);
  }finally{
    button.disabled = false;
    button.textContent = "Create account";
  }
};

document.querySelectorAll(".nav").forEach(b => b.onclick = () => navigate(b.dataset.page));
document.querySelectorAll("[data-page='help']").forEach(b => b.onclick = () => navigate("help"));
if($("#topNewScreening")) $("#topNewScreening").onclick = () => navigate("screening");
$("#heroScan").onclick = () => {
  navigate("screening");
  $("#resultPanel").hidden = true;
  $("#qualityMessage").innerHTML = "";
};
$("#historyBtn").onclick = $("#heroHistory").onclick = () => navigate("history");

$("#reportsDownload").onclick = () => {
  if(latestReportHtml){ downloadReport(); return; }
  if(window.latestStoredReportId){ downloadStoredReport(window.latestStoredReportId); return; }
  navigate("screening"); $("#qualityMessage").innerHTML = '<div class="quality-inline"><strong>No report available yet.</strong><span>Complete a screening to generate your first report.</span></div>';
};

function syncProfile(){
  const name=(user?.name||"").trim() || "Screening user";
  const email=(user?.email||"").trim() || "—";
  if($("#profileName")) $("#profileName").textContent=name;
  if($("#profileEmail")) $("#profileEmail").textContent=email;
  if($("#profileAvatar")) $("#profileAvatar").textContent=name.charAt(0).toUpperCase();
  if($("#profileNameDetail")) $("#profileNameDetail").textContent=name;
  if($("#profileEmailDetail")) $("#profileEmailDetail").textContent=email;
}

$("#logout").onclick = async () => {
  if(token){
    try{
      await fetch("/api/logout",{method:"POST",headers:{Authorization:"Bearer " + token}});
    }catch(_){}
  }
  localStorage.removeItem("ov_token");
  localStorage.removeItem("ov_user");
  token = null;
  user = null;
  selectedFile = null;
  $("#imageFile").value = "";
  $("#resultPanel").hidden = true;
  showAuth();
  $("#loginPassword").value = "";
};


if($("#historySearch")) $("#historySearch").addEventListener('input',()=>{
  const q=$("#historySearch").value.trim().toLowerCase();
  renderHistoryList((window.screeningHistory||[]).filter(x=>`${x.prediction||''} ${x.image_path||''}`.toLowerCase().includes(q)));
});
if($("#historyNewScreening")) $("#historyNewScreening").onclick=()=>navigate('screening');
if($("#reportsNewScreening")) $("#reportsNewScreening").onclick=()=>navigate('screening');
if($("#profileHistoryBtn")) $("#profileHistoryBtn").onclick=()=>navigate('history');
if($("#profileSignOut")) $("#profileSignOut").onclick=()=>$("#logout").click();
document.querySelectorAll('[data-close-modal]').forEach(el=>el.addEventListener('click',()=>{document.querySelectorAll('.modal').forEach(m=>m.hidden=true);}));
document.querySelectorAll('[data-help-target]').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.helpTarget)));
function openSupportModal(){ $("#supportModal").hidden=false; $("#supportSent").hidden=true; $("#supportCopied").hidden=true; $("#supportMessage").value=""; }
$("#openSupportBtn")?.addEventListener('click',openSupportModal); $("#openSupportBtn2")?.addEventListener('click',openSupportModal); $("#supportMessageCard")?.addEventListener('click',openSupportModal);
function supportPayload(){ return {message:$("#supportMessage")?.value.trim()||"", page:$("#pageContext")?.textContent||"OcuVisionAI workspace"}; }
$("#copySupportBtn")?.addEventListener('click',async()=>{ const p=supportPayload(); const msg=`OcuVisionAI support request\nAccount: ${user?.name||'Unknown'}\nEmail: ${user?.email||'Unknown'}\nPage: ${p.page}\nIssue: ${p.message||'Not provided'}\n\nPlease review and assist.`; try{await navigator.clipboard.writeText(msg); $("#supportCopied").hidden=false;}catch(_){alert(msg);} });
$("#sendSupportBtn")?.addEventListener('click',async()=>{ const p=supportPayload(); const b=$("#sendSupportBtn"); if(!p.message){ $("#supportSent").hidden=false; $("#supportSent").textContent="Please describe the issue before sending."; $("#supportSent").className="support-sent error"; return; } b.disabled=true; b.textContent="Sending..."; try{ const r=await fetch('/api/support',{method:'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+token},body:JSON.stringify(p)}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Unable to send support request.'); $("#supportSent").textContent=`Support request #${d.id} sent successfully. Your request has been recorded.`; $("#supportSent").className="support-sent success"; $("#supportSent").hidden=false; $("#supportMessage").value=""; }catch(err){ $("#supportSent").textContent=err.message; $("#supportSent").className="support-sent error"; $("#supportSent").hidden=false; }finally{ b.disabled=false; b.textContent="Send support request"; } });
$("#imageFile").onchange = (e) => {
  selectedFile = e.target.files[0] || null;
  const maxBytes = 200 * 1024 * 1024;
  if(selectedFile && selectedFile.size > maxBytes){
    selectedFile = null;
    e.target.value = "";
    $("#analyzeBtn").disabled = true;
    $("#clearFileBtn").hidden = true;
    $("#fileStatus").textContent = "File exceeds 200 MB";
    $("#qualityMessage").innerHTML = '<div class="quality-inline bad"><strong>File too large</strong><span>Please choose a JPG, JPEG or PNG image up to 200 MB.</span></div>';
    $("#filePreview").hidden = true;
    $("#filePreview").innerHTML = "";
    return;
  }
  $("#analyzeBtn").disabled = !selectedFile;
  $("#clearFileBtn").hidden = !selectedFile;
  $("#fileStatus").textContent = selectedFile ? selectedFile.name : "No image selected";
  $("#qualityMessage").innerHTML = "";
  $("#resultPanel").hidden = true;
  showFilePreview(selectedFile);
};

$("#clearFileBtn").onclick = () => {
  selectedFile = null;
  $("#imageFile").value = "";
  $("#fileStatus").textContent = "No image selected";
  $("#analyzeBtn").disabled = true;
  $("#clearFileBtn").hidden = true;
  $("#qualityMessage").innerHTML = "";
  $("#resultPanel").hidden = true;
  $("#filePreview").hidden = true;
  $("#filePreview").innerHTML = "";
};

function showFilePreview(file){
  const box = $("#filePreview");
  if(!file){ box.hidden = true; box.innerHTML = ""; return; }
  const sizeMB = file.size / (1024 * 1024);
  const reader = new FileReader();
  reader.onload = () => {
    box.innerHTML = `<img src="${reader.result}" alt="Selected retinal image preview"><div><strong>${escapeHtml(file.name)}</strong><span>${sizeMB.toFixed(2)} MB · ${escapeHtml(file.type || "image")}</span><small>Image selected and ready for quality assessment.</small></div>`;
    box.hidden = false;
  };
  reader.readAsDataURL(file);
}

$("#analyzeBtn").onclick = async (e) => {
  if(!selectedFile) return;
  if(selectedFile.size > 200 * 1024 * 1024){
    $("#qualityMessage").innerHTML = '<div class="quality-inline bad"><strong>File too large</strong><span>Please choose a JPG, JPEG or PNG image up to 200 MB.</span></div>';
    return;
  }
  const button = $("#analyzeBtn");
  button.disabled = true;
  button.textContent = "Analyzing image...";
  $("#qualityMessage").innerHTML = '<div class="quality-inline">Running image quality assessment and AI screening…</div>';

  const fd = new FormData();
  fd.append("file", selectedFile);

  try{
    const r = await fetch("/api/analyze", {
      method:"POST",
      headers:{Authorization:"Bearer " + token},
      body:fd
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || "Analysis failed.");

    const q = d.quality || {};
    const qClass = q.status === "Good" ? "good" : q.status === "Borderline" ? "borderline" : "bad";
    const qualityText = q.status === "Good"
      ? "Image quality is suitable for screening."
      : (q.failed_checks?.length ? q.failed_checks.join(" ") : (d.message || "Image quality needs review."));

    if(d.ok === false){
      $("#qualityMessage").innerHTML = "";
      $("#resultPanel").hidden = false;
      $("#resultPanel").innerHTML = renderPoorQuality(d);
      $("#resultPanel").scrollIntoView({behavior:"smooth", block:"start"});
      return;
    }

    $("#qualityMessage").innerHTML = "";
    $("#resultPanel").hidden = false;
    $("#resultPanel").innerHTML = renderResult(d);
    $("#resultPanel").scrollIntoView({behavior:"smooth", block:"start"});
    loadHistory();
    loadReports();
  }catch(err){
    $("#qualityMessage").innerHTML = `<div class="quality-inline bad">${escapeHtml(err.message)}</div>`;
  }finally{
    button.disabled = false;
    button.textContent = "Run AI screening";
  }
};

function renderPoorQuality(d){
  const q = d.quality || {};
  const reasons = (q.failed_checks || []).map(x => `<li>${escapeHtml(x)}</li>`).join("");
  return `
  <div class="poor-result poor-result-clean">
    <div class="poor-icon">!</div>
    <div>
      <span class="eyebrow danger-eyebrow">IMAGE QUALITY GATE</span>
      <h2>Image quality: Poor</h2>
      <p class="poor-lead">The retinal image is not suitable for reliable AI screening.</p>
      <div class="recapture-box">
        <strong>Recapture required</strong>
        <span>Capture a sharper fundus image with more uniform illumination. Keep the retinal field centered, complete and clearly visible, then upload the new image.</span>
      </div>
      ${reasons ? `<ul class="poor-reasons">${reasons}</ul>` : ""}
    </div>
  </div>`;
}

function renderResult(d){
  const probs = Object.entries(d.class_probabilities || {});
  const imgs = d.images || {};
  const flag = !!d.referable_flag;
  const q = d.quality || {};
  const ev = d.evidence_summary || {};
  const cal = d.confidence_calibration || {};
  latestReportHtml = d.report_html || "";
  window.latestStoredReportId = d.report_id || window.latestStoredReportId || null;

  setTimeout(() => {
    const reportBtn = $("#downloadReport");
    if(reportBtn) reportBtn.onclick = downloadReport;
  }, 0);

  const referralTitle = flag ? "Clinical review recommended" : "No Level 2+ screening flag";
  const referralText = flag
    ? "The AI screening result falls within the Level 2+ referable grouping. A qualified clinician should review the evidence and make the final referral decision."
    : "The AI screening result is below the Level 2+ screening-support threshold. A qualified clinician should interpret the result in clinical context.";
  const dotClass = flag ? "danger" : "success";

  return `
  <div class="result-shell">
    <div class="panel result-hero-panel result-hero-professional">
      <div class="result-header result-header-large">
        <div class="result-main-copy">
          <span class="eyebrow result-kicker">02 · AI SCREENING RESULT</span>
          <div class="result-heading-row"><h2 class="result-title result-title-large">${escapeHtml(d.prediction)}</h2><span class="result-severity-pill ${flag ? "attention" : "stable"}"><i></i>${flag ? "Review support" : "Screening result"}</span></div>
          <div class="confidence confidence-large">Model confidence <strong>${Number(d.confidence).toFixed(1)}%</strong></div>
          <div class="calibration-line"><span class="status-dot"></span>${escapeHtml(cal.label || "Confidence status available for review")}</div>
        </div>
        <div class="result-actions"><button id="downloadReport" class="secondary report-button">Download screening report</button></div>
      </div>
      <div class="result-disclaimer"><strong>AI screening support only.</strong> Model output and visual evidence are not a confirmed clinical diagnosis.</div>
    </div>

    <div class="panel quality-feature-panel">
      <div class="quality-status-banner ${q.status === "Good" ? "good" : q.status === "Borderline" ? "borderline" : "bad"}">
        <div><span class="quality-status-label">IMAGE QUALITY</span><strong>${escapeHtml(q.status || "Unknown")}</strong><span class="quality-score">QC score ${Number(q.quality_score || 0).toFixed(0)}/100</span></div>
        <p class="quality-main-message">${q.status === "Good" ? "Image quality is good and suitable for AI screening support." : (q.failed_checks?.join(" ") || "Image quality requires review.")}</p>
        <small class="quality-method">Focus, retinal illumination and usable field are assessed together.</small>
      </div>
      <div class="section-title"><span class="number">03</span><h3>Image quality assessment</h3></div>
      <div class="quality-grid quality-grid-large">
        ${qualityCard("Focus", q.focus_status, `Detail ${Number(q.sharpness || 0).toFixed(1)}`)}
        ${qualityCard("Illumination", q.illumination_status, `ROI mean ${Number(q.roi_mean || q.brightness || 0).toFixed(1)}`)}
        ${qualityCard("Field of view", q.field_status, `Coverage ${(Number(q.retinal_coverage || 0)*100).toFixed(0)}%`)}
      </div>
      ${q.status === "Borderline" && d.enhanced_quality ? `<div class="borderline-review"><div><strong>Borderline image — adaptive enhancement applied</strong><span>This screening was assessed independently for the selected image. Enhancement is used to improve visibility before AI review.</span></div><div class="borderline-after"><span>Post-enhancement quality</span><strong class="${d.enhanced_quality.status === "Good" ? "good" : d.enhanced_quality.status === "Borderline" ? "borderline" : "bad"}">${escapeHtml(d.enhanced_quality.status)}</strong></div></div>` : ""}
    </div>

    <div class="panel">
      <div class="section-title"><span class="number">04</span><h3>Original and enhanced retinal image</h3></div>
      <div class="image-grid image-grid-large">${imageCard(imgs.original,"Original retinal image")}${imageCard(imgs.enhanced,"Enhanced retinal image")}</div>
      <div class="evidence-note">Adaptive enhancement uses illumination normalization, CLAHE, denoising and mild sharpening. Enhancement supports visualization and does not guarantee clinical adequacy.</div>
    </div>

    <div class="panel probability-panel">
      <div class="section-title"><span class="number">05</span><h3>DR severity distribution</h3></div>
      <p class="section-lead">Model probability across the five trained severity classes.</p>
      <div class="prob-list prob-list-large">${probs.map(([name,value]) => `<div class="prob-row"><span>${escapeHtml(name)}</span><div class="bar"><i style="width:${Math.min(100,Math.max(0,Number(value)))}%"></i></div><strong>${Number(value).toFixed(1)}%</strong></div>`).join("")}</div>
    </div>

    <div class="panel xai-panel">
      <div class="section-title"><span class="number">06</span><h3>Explainable AI · model evidence</h3></div>
      <div class="xai-grid xai-grid-expanded">
        <figure class="image-card primary-xai primary-xai-large"><img src="${imgs.gradcam || imgs.original}" alt="Grad-CAM model evidence"><figcaption><strong>Grad-CAM attention map</strong><span>Regions that contributed to the predicted class</span></figcaption></figure>
        <div class="xai-copy">
          <span class="mini-label">CLINICIAN REVIEW CUE</span><h4>How to read the attention map</h4>
          <p>Grad-CAM highlights image regions that influenced the neural network output for the predicted class.</p>
          <div class="xai-points">
            <div class="xai-point"><i></i><span><strong>Model attention:</strong> identifies where the network placed emphasis.</span></div>
            <div class="xai-point"><i></i><span><strong>Cross-check:</strong> compare the heatmap with the original, enhanced and structural evidence.</span></div>
            <div class="xai-point"><i></i><span><strong>Clinical limitation:</strong> attention is not proof of a lesion, diagnosis or causal feature.</span></div>
          </div>
          <div class="xai-validation-card"><strong>Human-in-the-loop review</strong><span>The evidence is presented to support rapid clinician validation rather than replace clinical judgement.</span></div>
        </div>
      </div>
      <div class="xai-compare">${imageCard(imgs.original,"Original reference")}${imageCard(imgs.enhanced,"Enhanced reference")}</div>
    </div>

    <div class="panel evidence-panel-professional">
      <div class="section-title"><span class="number">07</span><h3>Retinal structures & lesion-level evidence</h3></div>
      <p class="section-lead">AI-assisted visual cues are separated into anatomy, localization, lesion candidates and vascular-change review groups.</p>
      <div class="evidence-group-title anatomy-title"><span class="evidence-group-marker anatomy"></span><div><strong>Retinal anatomy & localization</strong><small>Structural orientation for clinical review</small></div></div>
      <div class="evidence-grid evidence-grid-expanded evidence-grid-eight">
        ${evidenceCard(imgs.vessel,"Vessel structure","STRUCTURE","vessel")}
        ${evidenceCard(imgs.optic_disc,"Optic disc localization","LOCALIZATION","disc")}
        ${evidenceCard(imgs.fovea,"Estimated foveal cue","LOCALIZATION","fovea")}
        ${evidenceCard(imgs.enhanced,"Enhanced retinal reference","REFERENCE","reference")}
      </div>
      <div class="evidence-group-title lesion-title"><span class="evidence-group-marker lesions"></span><div><strong>Lesion candidate evidence</strong><small>Candidate regions for human review</small></div></div>
      <div class="evidence-grid evidence-grid-expanded evidence-grid-eight">
        ${evidenceCard(imgs.microaneurysm,"Microaneurysm candidates","CANDIDATE","microaneurysm")}
        ${evidenceCard(imgs.exudate,"Exudate candidates","CANDIDATE","exudate")}
        ${evidenceCard(imgs.hemorrhage,"Hemorrhage candidates","CANDIDATE","hemorrhage")}
        ${evidenceCard(imgs.lesion,"Combined lesion evidence","SUMMARY","combined")}
      </div>
      <div class="evidence-group-title vascular-title"><span class="evidence-group-marker vascular"></span><div><strong>Vascular change cue</strong><small>Fine-vessel pattern reviewed separately</small></div></div>
      <div class="evidence-grid evidence-grid-single">${evidenceCard(imgs.neovascularization,"Neovascularization candidates","VASCULAR CUE","neovascularization")}</div>
      <div class="candidate-summary-grid">${candidateStat("Microaneurysm candidates", ev.microaneurysm_candidates)}${candidateStat("Exudate candidates", ev.exudate_candidates)}${candidateStat("Hemorrhage candidates", ev.hemorrhage_candidates)}${candidateStat("Neovascularization candidates", ev.neovascularization_candidates)}</div>
      <div class="evidence-note strong-note">These visualizations are AI-assisted candidate evidence cues produced by image processing. They are not validated clinical segmentation or lesion-detection models and should not be interpreted as confirmed lesions.</div>
    </div>

    <div class="panel referral-panel referral-panel-large">
      <div class="section-title"><span class="number">08</span><h3>Referral support</h3></div>
      <div class="referral-card ${flag ? "flagged" : "clear"}">
        <div class="referral-status-line"><span class="referral-dot ${dotClass}"></span><div><div class="label">SCREENING-SUPPORT STATUS</div><div class="main">${referralTitle}</div><div class="sub">${referralText}</div></div></div>
        <span class="referral-status ${flag ? "flag" : "clear"}">${flag ? "REVIEW" : "NOT FLAGGED"}</span>
      </div>
      <div class="support-grid support-grid-large">
        <div class="support-card"><span>AI SEVERITY</span><strong>${escapeHtml(d.prediction)}</strong></div>
        <div class="support-card"><span>CONFIDENCE</span><strong>${Number(d.confidence).toFixed(1)}%</strong></div>
        <div class="support-card"><span>IMAGE QUALITY</span><strong>${escapeHtml(q.status || "—")}</strong></div>
        <div class="support-card"><span>DISC / FOVEA CUES</span><strong>${d.optic_disc_location ? "Available" : "Limited"}</strong></div>
      </div>
      <div class="notice warning"><span class="notice-dot"></span>Referral support is an AI screening aid. A qualified clinician should make the final referral or management decision.</div>
    </div>

    <div class="saved saved-large">Screening completed and saved to your OcuVisionAI history.</div>
  </div>`;
}

function candidateStat(label,value){
  const v = Number.isFinite(Number(value)) ? Number(value) : 0;
  return `<div class="candidate-stat"><span>${escapeHtml(label)}</span><strong>${v}</strong><small>candidate regions</small></div>`;
}

function downloadReport(){
  if(!latestReportHtml) return;
  const blob = new Blob([latestReportHtml], {type:"text/html;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "OcuVisionAI_Retinal_Screening_Report.html";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function evidenceCard(url,label,chip,type="default"){
  if(!url) return "";
  const desc = evidenceDescription(type);
  const visual = evidenceVisual(type);
  return `<figure class="image-card evidence-card-${escapeHtml(type)}">
    <div class="evidence-image-wrap"><img src="${url}" alt="${escapeHtml(label)}"></div>
    <div class="evidence-meta">
      <div class="evidence-title-row"><span class="evidence-icon ${escapeHtml(type)}">${visual.icon}</span><strong>${escapeHtml(label)}</strong></div>
      <small>${escapeHtml(desc)}</small>
      <span class="evidence-chip">${escapeHtml(chip)}</span>
    </div>
  </figure>`;
}

function evidenceDescription(type){
  const map={
    vessel:"Retinal vessel pattern",
    disc:"Optic disc cue",
    fovea:"Estimated macular cue",
    reference:"Processed reference",
    microaneurysm:"Small focal candidate regions",
    exudate:"Bright lesion candidate regions",
    hemorrhage:"Dark/red candidate regions",
    combined:"Combined lesion evidence view",
    neovascularization:"Fine-vessel change cue"
  };
  return map[type] || "AI-assisted review cue";
}

function evidenceVisual(type){
  const map={
    vessel:{icon:"V",name:"structure"},
    disc:{icon:"D",name:"localization"},
    fovea:{icon:"F",name:"localization"},
    reference:{icon:"R",name:"reference"},
    microaneurysm:{icon:"M",name:"lesion"},
    exudate:{icon:"E",name:"bright"},
    hemorrhage:{icon:"H",name:"blood"},
    combined:{icon:"+",name:"summary"},
    neovascularization:{icon:"N",name:"vascular"}
  };
  return map[type] || {icon:"•",name:"evidence"};
}

function qualityCard(label,status,metric){
  const cls = status === "Good" ? "good" : status === "Borderline" ? "borderline" : "bad";
  return `<div class="quality-card"><span>${label}</span><strong class="${cls}">${status}</strong><div class="metric">${escapeHtml(String(metric))}</div></div>`;
}

function imageCard(url,label){
  if(!url) return "";
  return `<figure class="image-card"><img src="${url}" alt="${escapeHtml(label)}"><figcaption>${escapeHtml(label)}</figcaption></figure>`;
}

async function loadHistory(){
  if(!token) return;
  try{
    const r = await fetch("/api/history",{headers:{Authorization:"Bearer " + token}});
    if(!r.ok) return;
    const d = await r.json();
    const h = d.history || [];
    $("#totalScans").textContent = h.length;
    $("#latestResult").textContent = h[0]?.prediction || "—";
    const latest=h[0] || null;
    const latestConf=latest && Number.isFinite(Number(latest.confidence)) ? `${Number(latest.confidence).toFixed(1)}%` : "—";
    $("#latestConfidenceValue").textContent = latest ? latestConf : "—";
    $("#latestConfidence").textContent = latest && latestConf !== "—" ? "Model confidence" : (latest ? "Confidence not recorded" : "No screening yet");
    $("#latestDate").textContent = latest ? formatShortDate(latest.created_at) : "—";
    if($("#latestDate")) $("#latestDate").title = latest ? formatDateTime(latest.created_at) : "No screening yet";

    window.screeningHistory = h;
    $("#recent").innerHTML = h.slice(0,5).map(historyRow).join("") || '<div class="empty-inline"><strong>No screenings yet</strong><span>Start a new screening to build your history.</span></div>';
    renderHistoryList(h);
    $("#profileTotal") && ($("#profileTotal").textContent = h.length);
    renderDashboardCharts(h);
    loadReports();
    syncProfile();
  }catch(_){}
}

function renderDashboardCharts(history){
  const confidenceEl = $("#confidenceChart");
  const diagnosisEl = $("#diagnosisChart");
  if(!confidenceEl || !diagnosisEl) return;

  if(!history.length){
    confidenceEl.innerHTML = '<div class="chart-empty"><strong>No screening data yet</strong><span>Complete a screening to see confidence trends here.</span></div>';
    diagnosisEl.innerHTML = '<div class="chart-empty"><strong>No results yet</strong><span>Your diagnosis distribution will appear after screening.</span></div>';
    return;
  }

  const recent = history.slice(0,8).reverse();
  const values = recent.map(x => Math.max(0, Math.min(100, Number(x.confidence || 0))));
  const W=640,H=250,padL=42,padR=20,padT=20,padB=36;
  const plotW=W-padL-padR, plotH=H-padT-padB;
  const xAt=i => recent.length===1 ? padL+plotW/2 : padL+(i/(recent.length-1))*plotW;
  const yAt=v => padT+(1-v/100)*plotH;
  const points=values.map((v,i)=>`${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ');
  const grid=[0,25,50,75,100].map(v=>{
    const y=yAt(v);
    return `<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" class="chart-grid-line"/><text x="${padL-10}" y="${y+4}" text-anchor="end" class="chart-axis">${v}%</text>`;
  }).join('');
  const circles=values.map((v,i)=>`<circle cx="${xAt(i)}" cy="${yAt(v)}" r="5" class="chart-point"><title>${v.toFixed(1)}% confidence</title></circle>`).join('');
  confidenceEl.innerHTML=`<svg class="confidence-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Confidence over time">${grid}<polyline points="${points}" class="chart-line"/>${circles}<text x="${padL}" y="${H-10}" class="chart-axis">Earlier</text><text x="${W-padR}" y="${H-10}" text-anchor="end" class="chart-axis">Latest</text></svg><div class="chart-footer"><span>Latest confidence <strong>${values[values.length-1].toFixed(1)}%</strong></span><span>${recent.length} screening${recent.length===1?'':'s'} shown</span></div>`;

  const counts={"No DR":0,"Mild":0,"Moderate":0,"Severe":0,"Proliferative DR":0};
  history.forEach(x=>{ if(counts[x.prediction] !== undefined) counts[x.prediction]++; });
  const entries=Object.entries(counts).filter(([,v])=>v>0);
  const total=history.length;
  const colors={"No DR":"#1d8a72","Mild":"#5b9bd5","Moderate":"#d49b38","Severe":"#d46a4f","Proliferative DR":"#8c4f9e"};
  diagnosisEl.innerHTML=`<div class="diagnosis-bars">${entries.map(([name,count])=>{const pct=(count/total)*100;return `<div class="diagnosis-row"><div class="diagnosis-label"><span class="diagnosis-dot" style="background:${colors[name]}"></span><strong>${escapeHtml(name)}</strong><span>${count}</span></div><div class="diagnosis-track"><i style="width:${pct}%;background:${colors[name]}"></i></div></div>`}).join('')}</div><div class="chart-footer"><span><strong>${total}</strong> total screening${total===1?'':'s'}</span><span>Distribution from saved history</span></div>`;
}

function historyRow(x){
  const prediction = x.prediction || "Unknown";
  const levelClass = prediction.toLowerCase().replace(/\s+/g,"-");
  const report = x.report_available ? `<button class="row-action report-row-btn" data-report-id="${escapeHtml(String(x.id || ""))}">Download report</button>` : `<span class="report-unavailable">Older entry</span>`;
  return `<div class="record history-record" data-history-id="${escapeHtml(String(x.id || ""))}" data-search="${escapeHtml(`${prediction} ${x.image_path || ""}`.toLowerCase())}">
    <div class="record-result"><span class="record-dot ${escapeHtml(levelClass)}"></span><div><strong>${escapeHtml(prediction)}</strong><small>${escapeHtml(x.image_path || "Retinal image")}</small></div></div>
    <span class="date">${escapeHtml(formatDateTime(x.created_at))}</span>
    <span class="conf"><b>${Number.isFinite(Number(x.confidence)) ? Number(x.confidence).toFixed(1)+"%" : "—"}</b><small>confidence</small></span>
    <span>${report}</span>
  </div>`;
}

function renderHistoryList(history){
  const list=$("#historyList");
  if(!list) return;
  list.innerHTML=history.map(historyRow).join("") || '<div class="empty-inline"><strong>No screening history</strong><span>Completed screenings will appear here.</span></div>';
  if($("#historyCount")) $("#historyCount").textContent=`${history.length} screening${history.length===1?'':'s'}`;
  list.querySelectorAll('.history-record').forEach(row=>row.onclick=(e)=>{ if(e.target.closest('button')) return; const id=String(row.dataset.historyId); const item=(window.screeningHistory||[]).find(x=>String(x.id)===id); if(item) showHistoryDetail(item); });
  list.querySelectorAll('.report-row-btn').forEach(btn=>btn.onclick=(e)=>{ e.stopPropagation(); downloadStoredReport(btn.dataset.reportId); });
}

async function showHistoryDetail(item){
  const modal=$("#historyDetailModal");
  if(!modal) return;
  $("#historyDetailTitle").textContent=item.prediction || "Screening";
  $("#historyDetailBody").innerHTML='<div class="detail-loading">Loading complete screening details…</div>';
  modal.hidden=false;
  try{
    const r=await fetch(`/api/history/${encodeURIComponent(item.id)}`,{headers:{Authorization:'Bearer '+token}});
    const d=await r.json();
    if(!r.ok) throw new Error(d.detail||'Unable to load screening details.');
    const a=d.analysis||{};
    const q=a.quality||{}; const eq=a.enhanced_quality||{}; const probs=a.class_probabilities||{}; const ev=a.evidence_summary||{};
    const probRows=Object.entries(probs).map(([k,v])=>`<div class="detail-prob"><span>${escapeHtml(k)}</span><strong>${Number(v).toFixed(1)}%</strong></div>`).join('');
    const imageCards=[['original','Original image'],['enhanced','Enhanced image'],['gradcam','Grad-CAM']].filter(([k])=>a.images?.[k]).map(([k,l])=>`<figure class="detail-image"><img src="${a.images[k]}" alt="${escapeHtml(l)}"><figcaption>${escapeHtml(l)}</figcaption></figure>`).join('');
    const reportBtn=d.report_available?`<button class="primary full-width-btn" onclick="downloadStoredReport('${escapeHtml(String(item.id))}')">Download complete screening report</button>`:'';
    $("#historyDetailBody").innerHTML=`
      <div class="detail-summary"><div><span>Screening ID</span><strong>#${escapeHtml(String(item.id||'—'))}</strong></div><div><span>Date &amp; time</span><strong>${escapeHtml(formatDateTime(item.created_at))}</strong></div><div><span>AI screening result</span><strong>${escapeHtml(a.prediction||item.prediction||'—')}</strong></div><div><span>Confidence</span><strong>${Number.isFinite(Number(a.confidence))?Number(a.confidence).toFixed(1)+'%':'—'}</strong></div></div>
      <div class="detail-section"><span class="eyebrow">IMAGE QUALITY</span><div class="detail-grid"><div><span>Focus</span><strong>${escapeHtml(q.focus_status||'—')}</strong></div><div><span>Illumination</span><strong>${escapeHtml(q.illumination_status||'—')}</strong></div><div><span>Field of view</span><strong>${escapeHtml(q.field_status||'—')}</strong></div><div><span>Post-enhancement</span><strong>${escapeHtml(eq.status||'Not applicable')}</strong></div></div></div>
      <div class="detail-section"><span class="eyebrow">CLASS PROBABILITIES</span><div class="detail-prob-list">${probRows||'<span>No probability data stored.</span>'}</div></div>
      <div class="detail-section"><span class="eyebrow">RETINAL EVIDENCE</span><div class="detail-grid"><div><span>Microaneurysm candidates</span><strong>${escapeHtml(ev.microaneurysm_candidates??'—')}</strong></div><div><span>Exudate candidates</span><strong>${escapeHtml(ev.exudate_candidates??'—')}</strong></div><div><span>Hemorrhage candidates</span><strong>${escapeHtml(ev.hemorrhage_candidates??'—')}</strong></div><div><span>Neovascularization candidates</span><strong>${escapeHtml(ev.neovascularization_candidates??'—')}</strong></div></div></div>
      ${imageCards?`<div class="detail-section"><span class="eyebrow">VISUAL REVIEW</span><div class="detail-images">${imageCards}</div></div>`:''}
      <div class="detail-note"><strong>Clinical-use notice</strong><p>These saved results are AI-assisted screening support. Candidate lesion and structure evidence is not confirmed clinical diagnosis.</p></div>${reportBtn||'<div class="notice warning">A complete report was not stored for this older screening entry.</div>'}`;
  }catch(err){ $("#historyDetailBody").innerHTML=`<div class="notice warning">${escapeHtml(err.message)}</div>`; }
}

async function loadReports(){
  if(!token) return;
  try{
    const r=await fetch('/api/reports',{headers:{Authorization:'Bearer '+token}});
    if(!r.ok) return;
    const d=await r.json();
    const reports=d.reports||[];
    if($("#profileReports")) $("#profileReports").textContent=reports.length;
    const list=$("#reportList");
    if(!list) return;
    list.innerHTML=reports.length?reports.map(r=>`<div class="report-row"><div class="report-row-icon"><svg viewBox="0 0 24 24"><path d="M6 3h9l4 4v14H6z"/><path d="M15 3v5h5M9 13h6M9 17h6"/></svg></div><div class="report-row-main"><strong>${escapeHtml(r.prediction || 'Screening report')}</strong><span>${escapeHtml(formatDateTime(r.created_at))} · ${Number.isFinite(Number(r.confidence))?Number(r.confidence).toFixed(1)+'% confidence':''}</span></div><div class="report-row-actions"><button class="secondary row-action" data-view-report-id="${escapeHtml(String(r.id))}">View</button><button class="primary row-action" data-report-id="${escapeHtml(String(r.id))}">Download</button></div></div>`).join(''):'<div class="empty-state"><strong>No report available yet</strong><span>Complete a screening to generate a structured report.</span></div>';
    list.querySelectorAll('[data-report-id]').forEach(b=>b.onclick=()=>downloadStoredReport(b.dataset.reportId));
    list.querySelectorAll('[data-view-report-id]').forEach(b=>b.onclick=()=>viewStoredReport(b.dataset.viewReportId));
    if($("#reportsStatus")) $("#reportsStatus").textContent=reports.length?`${reports.length} report${reports.length===1?'':'s'} available`:'No report selected';
    const latest=reports[0];
    if(latest) window.latestStoredReportId=latest.id;
  }catch(_){}
}

async function viewStoredReport(id){
  if(!id) return;
  try{
    const r=await fetch(`/api/reports/${encodeURIComponent(id)}`,{headers:{Authorization:'Bearer '+token}});
    if(!r.ok){ const d=await r.json().catch(()=>({})); throw new Error(d.detail||'Report is not available.'); }
    const d=await r.json();
    const blob=new Blob([d.report_html||''],{type:'text/html;charset=utf-8'});
    const url=URL.createObjectURL(blob);
    window.open(url,'_blank','noopener,noreferrer');
    setTimeout(()=>URL.revokeObjectURL(url),60000);
  }catch(err){ alert(err.message); }
}

async function downloadStoredReport(id){
  if(!id) return;
  try{
    const r=await fetch(`/api/reports/${encodeURIComponent(id)}/download`,{headers:{Authorization:'Bearer '+token}});
    if(!r.ok){ const d=await r.json().catch(()=>({})); throw new Error(d.detail||'Report is not available.'); }
    const blob=await r.blob(); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=`OcuVisionAI_Screening_Report_${id}.html`; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(err){ alert(err.message); }
}

function formatShortDate(value){
  if(!value) return "—";
  const d = new Date(String(value).replace(" ","T"));
  if(Number.isNaN(d.getTime())) return String(value).slice(0,10);
  return d.toLocaleDateString(undefined,{day:"2-digit",month:"short"});
}

function formatDateTime(value){
  if(!value) return "—";
  const d = new Date(String(value).replace(" ","T"));
  if(Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleDateString(undefined,{day:"2-digit",month:"short",year:"numeric"}) + " · " + d.toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"});
}

function escapeHtml(value){
  return String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
}

if(token && user) showApp();
else showAuth();
