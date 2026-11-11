const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const pct = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : '—';
const num = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const names = {logistic_regression:'逻辑回归',decision_tree:'决策树',kmeans:'K-Means',agglomerative:'层次聚类',pca:'PCA',truncated_svd:'Truncated SVD',quality_report:'质量报告',hadoop_streaming:'Hadoop 清洗',experiment_history:'实验记录'};
const taskNames = {classification:'高分预测',clustering:'用户分群',reduction:'用户画像降维',quality:'数据质量',history:'实验历史'};
const palette = ['#2d7c9a','#e1a466','#74a883','#9c83b5','#bf6f7b','#7d93bd','#a6a060','#619e9c'];
let state = {sessionId:null, session:null, sessions:[], jobId:null, jobSessionId:null, trainId:null, bootstrap:null};

async function api(path, options = {}) {
  const response = await fetch(path, {headers: {'Content-Type':'application/json'}, ...options});
  let data;
  try { data = await response.json(); } catch { throw new Error(`请求失败：HTTP ${response.status}`); }
  if (!response.ok) throw new Error(data.error || `请求失败：HTTP ${response.status}`);
  return data;
}

function toast(message) {
  const box = $('#toast'); box.textContent = message; box.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => box.hidden = true, 4800);
}

function renderRail() {
  $('#session-count').textContent = state.sessions.length;
  $('#session-list').innerHTML = state.sessions.map(item =>
    `<button class="session-item ${item.id === state.sessionId ? 'active' : ''}" data-session="${escapeHTML(item.id)}" title="${escapeHTML(item.title)}">${escapeHTML(item.title || '新对话')}</button>`
  ).join('');
  $('#session-list').querySelectorAll('[data-session]').forEach(button => button.addEventListener('click', () => openSession(button.dataset.session)));
}

function fileSize(bytes) { return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`; }

function renderInspector() {
  const b = state.bootstrap;
  if (!b) return;
  $('#model-status').textContent = b.model_ready ? '模型已就绪' : '模型需要训练';
  $('#model-status').classList.toggle('warning', !b.model_ready);
  $('#data-status').innerHTML = `
    <div class="data-line"><span>Hadoop 清洗</span><strong>${b.quality ? '已登记' : '未运行'}</strong></div>
    <div class="data-line"><span>六种算法</span><strong>${b.model_ready ? '可查询' : '待训练'}</strong></div>
    <div class="data-line"><span>DeepSeek 路由</span><strong>${b.deepseek_configured ? '已配置' : '未配置'}</strong></div>
    ${b.quality?.split_counts ? `<div class="data-version">训练 / 验证 / 测试：${Object.values(b.quality.split_counts).map(value => Number(value).toLocaleString()).join(' / ')}</div>` : ''}
    ${b.quality?.boundaries?.T1 ? `<div class="data-version">T1 ${escapeHTML(b.quality.boundaries.T1)}<br>T2 ${escapeHTML(b.quality.boundaries.T2)}</div>` : ''}
    ${b.experiment ? `<div class="data-version">实验 ${escapeHTML(b.experiment.experiment_id)}</div>` : ''}
    ${b.quality ? `<div class="data-version">数据版本 ${escapeHTML(b.quality.data_version)}</div>` : ''}`;
  const files = b.artifacts || [];
  $('#artifact-list').innerHTML = files.length ? files.map((file, index) => `
    ${index === 0 ? '<div class="artifact-group">清洗产物</div>' : index === 4 ? '<div class="artifact-group">模型产物</div>' : ''}
    <a class="artifact-link" href="/api/artifacts/${encodeURIComponent(file.id)}" download>
      <span class="file-icon">${file.filename.endsWith('.json') ? '{}' : '≡'}</span><span class="file-label">${escapeHTML(file.label)}</span><span class="size">${fileSize(file.size)}</span>
    </a>`).join('') : '<p class="side-copy">运行清洗和训练后，可在这里下载报告与结果。</p>';
}

function welcome() {
  const examples = [
    ['检查数据质量','对照清洗前后的五维质量评分','查看数据质量及清洗结果'],
    ['预测高分','UserID 1 对 MovieID 1 会打高分吗','预测 UserID 1 对 MovieID 1 是否打高分'],
    ['比较用户群体','查看两种聚类算法的用户画像','比较 K-Means 与层次聚类的用户群体'],
    ['查看二维坐标','比较 PCA 与 SVD 的降维结果','比较 PCA 与 SVD 的二维降维结果']
  ];
  return `<section class="welcome"><div class="welcome-overline">MovieLens 1M · 数据分析 Agent</div><h2>从一个问题开始分析电影评分</h2>
    <p>用自然语言调用已训练的分类、聚类和降维工具。结果展示真实指标、数据版本与可下载文件。</p>
    <div class="prompt-grid">${examples.map(([title,sub,prompt]) => `<button type="button" class="suggestion" data-prompt="${escapeHTML(prompt)}"><strong>${escapeHTML(title)}</strong><span>${escapeHTML(sub)}</span></button>`).join('')}</div>
    <div class="welcome-foot">也可以在下方输入 UserID、MovieID 或具体算法名称。</div></section>`;
}

function bar(value) { return `<div class="bar"><i style="width:${Math.max(0,Math.min(100,Number(value)*100))}%"></i></div>`; }
function metric(label, value) { return `<div class="metric-line"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`; }
function detail(data, label = '查看原始工具结果') { return `<details class="details"><summary>${label}</summary><pre>${escapeHTML(JSON.stringify(data, null, 2))}</pre></details>`; }

function renderQuality(report) {
  const quality = report.quality || {};
  if (!quality.before || !quality.after) return detail(report);
  const labels = {'Accurate':'准确性','Complete':'完整性','Unique':'唯一性','Up-to-date':'时效性','Consistent':'一致性'};
  const rejected = ['users','movies','ratings'].reduce((sum,key) => sum + Number(report.cleaning?.[key]?.rejected || 0), 0);
  return `<div class="result-title">清洗前后质量对照</div>
    <div class="result-band"><div><div class="label">五维质量评估</div><div class="value">原始数据 → 清洗数据</div></div><div class="note">隔离 ${rejected.toLocaleString()} 条异常记录<br>规则 ${escapeHTML(report.rule_version || '—')}</div></div>
    <table class="data-table"><thead><tr><th>质量维度</th><th>清洗前</th><th>清洗后</th><th>变化</th></tr></thead><tbody>
    ${Object.keys(quality.before).map(key => `<tr><td>${labels[key] || escapeHTML(key)}</td><td>${num(quality.before[key],2)}</td><td>${num(quality.after[key],2)}</td><td>+${num(quality.after[key]-quality.before[key],2)}</td></tr>`).join('')}</tbody></table>
    <p class="small-note">评分来自 Hadoop Streaming 对原始数据和清洗数据使用同一规则的评价。隔离记录未被修复。</p>
    ${detail({cleaning:report.cleaning,quality_method:quality.method,boundaries:report.boundaries},'查看隔离原因、评分口径与时间边界')}`;
}

function renderClassification(results) {
  const first = results[0] || {};
  const prediction = first.positive_probability !== undefined;
  return `<div class="result-title">两种分类模型${prediction ? '的高分预测' : '的测试指标'}</div>
    ${prediction ? `<div class="result-band"><div><div class="label">预测电影</div><div class="value">${escapeHTML(first.movie_title)}</div></div><div class="note">UserID ${escapeHTML(first.UserID)} · MovieID ${escapeHTML(first.MovieID)}<br>高分定义：评分 ≥ 4</div></div>` : ''}
    <div class="metric-pair">${results.map(item => { const test = item.test_metrics || item.test || {};
      return `<div class="metric-box"><h4>${escapeHTML(names[item.tool] || item.tool)}</h4>
        ${prediction ? `${metric('高分概率',pct(item.positive_probability))}${bar(item.positive_probability)}${metric('预测判断',item.predicted_high_rating ? '倾向高分' : '倾向非高分')}${metric('判断阈值',pct(item.threshold))}` : ''}
        ${metric('测试集 Precision',num(test.precision))}${metric('测试集 Recall',num(test.recall))}
        ${metric('测试集 F1',num(test.f1))}${metric('测试集 ROC-AUC',num(test.roc_auc))}
        <div class="small-note">${escapeHTML(item.parameter || '')}</div></div>`; }).join('')}</div>
    ${first.target_pair_in_training ? '<div class="warning-note">目标用户与电影组合曾出现在训练期；本次预测已从历史聚合特征中扣除该组合评分，不能当作样本外准确率。</div>' : ''}
    ${first.history_warning ? `<div class="warning-note">${escapeHTML(first.history_warning)}</div>` : ''}
    <p class="small-note">测试集指标用于评价模型；单次预测仅对应当前用户与电影。</p>${detail(results)}`;
}

function renderClustering(results) {
  const first = results[0] || {};
  const groups = first.groups || [];
  return `<div class="result-title">用户分群结果</div>
    <div class="result-band"><div><div class="label">训练期用户画像</div><div class="value">${Number(first.users || 0).toLocaleString()} 位用户 · ${escapeHTML(first.n_clusters)} 个群体</div></div><div class="note">${results.length} 种聚类方法<br>当前实验的训练期用户</div></div>
    <div class="metric-pair">${results.map(item => `<div class="metric-box"><h4>${escapeHTML(names[item.tool] || item.tool)}</h4>${metric('抽样轮廓系数',num(item.silhouette,5))}${metric('群体数量',item.n_clusters)}${item.UserID ? metric(`UserID ${item.UserID} 属于`,`${item.cluster} 群`) : ''}</div>`).join('')}</div>
    ${first.selected_users ? `<table class="data-table"><thead><tr><th>UserID</th><th>群体</th><th>训练期评分数</th></tr></thead><tbody>${first.selected_users.map(row => `<tr><td>${escapeHTML(row.UserID)}</td><td>${escapeHTML(row.cluster)}</td><td>${escapeHTML(row.rating_count)}</td></tr>`).join('')}</tbody></table><p class="small-note">所选用户的群体来自全体训练期模型，没有对该子集重新聚类。</p>` : ''}
    ${results.map(item => item.history_warning ? `<div class="warning-note">${escapeHTML(item.history_warning)}</div>` : '').join('')}
    ${groups.length ? `<div class="result-title">${escapeHTML(names[first.tool] || first.tool)} 群体画像</div><table class="data-table"><thead><tr><th>群体</th><th>用户数</th><th>平均评分</th><th>相对偏好</th><th>代表用户</th></tr></thead><tbody>${groups.map(row => `<tr><td>${escapeHTML(row.cluster)}</td><td>${Number(row.users).toLocaleString()}</td><td>${num(row.mean_rating,2)}</td><td>${escapeHTML(row.explanation || '—')}</td><td>${escapeHTML((row.representative_user_ids || []).join('、'))}</td></tr>`).join('')}</tbody></table>` : ''}
    <p class="small-note">类型偏好基于训练期评分统计，仅描述已有用户群体。</p>${detail(results)}`;
}

function renderReduction(results, suffix) {
  const first = results[0] || {};
  const clusterIds = [...new Set((first.coordinates_sample || []).map(row => Number(row.kmeans)))].sort((a,b) => a-b);
  return `<div class="result-title">二维用户画像</div>
    <div class="result-band"><div><div class="label">降维结果</div><div class="value">${escapeHTML(first.original_dimensions)} 维 → ${escapeHTML(first.components)} 维</div></div><div class="note">${Number(first.coordinate_count || 0).toLocaleString()} 位用户的坐标<br>颜色表示训练期 K-Means 群体</div></div>
    <div class="metric-pair">${results.map(item => `<div class="metric-box"><h4>${escapeHTML(names[item.tool] || item.tool)}</h4>${metric('解释方差比例',pct(item.explained_variance_ratio_sum))}${bar(item.explained_variance_ratio_sum)}${metric('重构均方误差',num(item.reconstruction_mse,5))}${metric('变换耗时',`${num(item.transform_seconds,3)} 秒`)}${metric('矩阵 → 坐标数组',`${fileSize(item.input_matrix_bytes || 0)} → ${fileSize(item.coordinate_bytes || 0)}`)}</div>`).join('')}</div>
    ${first.retention_threshold !== undefined ? `<div class="${first.retention_met ? 'small-note' : 'warning-note'}">要求信息保留至少 ${pct(first.retention_threshold)}；当前二维模型${first.retention_met ? '已达到' : '未达到'}。${first.retention_met ? '' : '需要重新训练更高维数的实验后再核对。'}</div>` : ''}
    ${first.selected_users ? `<div class="result-title">指定用户的降维坐标</div><table class="data-table"><thead><tr><th>UserID</th><th>坐标 1</th><th>坐标 2</th><th>K-Means</th><th>层次聚类</th></tr></thead><tbody>${first.selected_users.map(row => `<tr><td>${escapeHTML(row.UserID)}</td><td>${num(row.component_1)}</td><td>${num(row.component_2)}</td><td>${escapeHTML(row.kmeans)}</td><td>${escapeHTML(row.agglomerative)}</td></tr>`).join('')}</tbody></table><p class="small-note">${escapeHTML(first.subset_note)}</p>` : ''}
    <div class="plot-grid">${results.map((item,index) => `<div class="plot-box"><h4>${escapeHTML(names[item.tool] || item.tool)} 用户分布</h4><div class="plot-sub">展示 ${Number((item.coordinates_sample || []).length).toLocaleString()} 个真实坐标</div><canvas id="plot-${suffix}-${index}" aria-label="${escapeHTML(names[item.tool] || item.tool)} 二维散点图"></canvas><div class="legend">${clusterIds.map((x) => `<span><i style="background:${palette[x%palette.length]}"></i>群体 ${x}</span>`).join('')}</div></div>`).join('')}</div>
    <div class="result-title">降维前后聚类对照</div>
    <table class="data-table"><thead><tr><th>降维</th><th>聚类</th><th>${escapeHTML(first.original_dimensions)} 维轮廓系数</th><th>${escapeHTML(first.components)} 维轮廓系数</th><th>原空间 / 降维后耗时</th></tr></thead><tbody>
    ${results.flatMap(item => Object.entries(item.comparison || {}).map(([algorithm,row]) => `<tr><td>${escapeHTML(names[item.tool] || item.tool)}</td><td>${escapeHTML(names[algorithm] || algorithm)}${item.comparison_groups?.[algorithm] ? `，${escapeHTML(item.comparison_groups[algorithm])} 群` : ''}</td><td>${num(row.original_silhouette,5)}</td><td>${num(row.reduced_silhouette,5)}</td><td>${num(row.original_cluster_seconds,3)} / ${num(row.reduced_cluster_seconds,3)} 秒</td></tr>`)).join('')}</tbody></table>
    <p class="small-note">原空间与二维空间的轮廓系数分别按各自距离空间计算，不代表相同量纲的信息保留率。</p>
    ${detail(results.map(({coordinates_sample,...other}) => other),'查看方法、资源用量与聚类对照')}`;
}

function renderHistory(results) {
  return `<div class="result-title">已登记实验</div><table class="data-table"><thead><tr><th>实验</th><th>维数</th><th>逻辑回归 F1</th><th>K-Means 轮廓系数</th></tr></thead><tbody>${results.map(item => `<tr><td>${escapeHTML(item.experiment_id)}</td><td>${escapeHTML(item.components)}</td><td>${num(item.logistic_test_f1)}</td><td>${num(item.kmeans_silhouette,5)}</td></tr>`).join('')}</tbody></table>${detail(results)}`;
}

function renderAnswer(answer, index) {
  const ok = answer.status === 'completed';
  const followup = Boolean(answer.explanation && answer.summary?.startsWith('继续解释上一条'));
  const toolNames = (answer.tools || []).map(tool => names[tool] || tool).join('与');
  let summary = answer.error || answer.summary || '';
  if (ok && !followup && answer.task === 'classification') summary = answer.results?.[0]?.positive_probability !== undefined ? `已读取${toolNames}对这位用户和电影的预测。` : `已读取${toolNames}的测试集表现。`;
  if (ok && !followup && answer.task === 'clustering') summary = `已读取${toolNames}的训练期用户分群。`;
  if (ok && !followup && answer.task === 'reduction') summary = `已读取当前实验中的${toolNames}降维结果。`;
  let body = '';
  if (ok && answer.results?.length && !followup) {
    if (answer.task === 'quality') body = renderQuality(answer.results[0]);
    else if (answer.task === 'classification') body = renderClassification(answer.results);
    else if (answer.task === 'clustering') body = renderClustering(answer.results);
    else if (answer.task === 'reduction') body = renderReduction(answer.results, index);
    else if (answer.task === 'history') body = renderHistory(answer.results);
  }
  return `<article class="message"><div class="message-meta"><span class="avatar">M</span><span>MovieLens Agent</span></div><div class="answer">
    <div class="trace"><span class="${ok ? 'good' : 'bad'}">${ok ? '分析完成' : '分析未完成'}</span>
    ${answer.task && answer.task !== 'unknown' ? `<span>${escapeHTML(taskNames[answer.task] || answer.task)}</span>` : ''}
    ${(answer.tools || []).map(tool => `<span>${escapeHTML(names[tool] || tool)}</span>`).join('')}
    ${answer.routing_source === 'deepseek' ? '<span>DeepSeek 意图识别</span>' : ''}</div>
    ${!followup ? `<p class="answer-summary">${escapeHTML(summary)}</p>` : ''}
    ${answer.explanation ? `<p class="answer-summary">${escapeHTML(answer.explanation)}</p>` : ''}${body}
    ${answer.experiment_id && !followup ? `<p class="small-note">实验 ${escapeHTML(answer.experiment_id)} · 数据版本 ${escapeHTML(answer.data_version)}</p>` : ''}</div></article>`;
}

function drawPlot(canvas, rows) {
  if (!canvas || !rows?.length) return;
  const width = canvas.clientWidth, height = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const values = rows.filter(row => Number.isFinite(Number(row.component_1)) && Number.isFinite(Number(row.component_2)));
  if (!values.length) return;
  const xs = values.map(x=>Number(x.component_1)), ys = values.map(x=>Number(x.component_2));
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const pad=20, w=width-2*pad,h=height-2*pad;
  ctx.strokeStyle='#d9e4eb';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad,height-pad);ctx.lineTo(width-pad,height-pad);ctx.moveTo(pad,pad);ctx.lineTo(pad,height-pad);ctx.stroke();
  for (const row of values) {
    const x=pad+(Number(row.component_1)-xmin)/((xmax-xmin)||1)*w;
    const y=height-pad-(Number(row.component_2)-ymin)/((ymax-ymin)||1)*h;
    ctx.fillStyle=palette[Number(row.kmeans)%palette.length] || palette[0];ctx.globalAlpha=.64;
    ctx.beginPath();ctx.arc(x,y,2.1,0,Math.PI*2);ctx.fill();
  }
  ctx.globalAlpha=1;
}

function renderConversation() {
  const messages = state.session?.messages || [];
  $('#conversation-title').textContent = state.session?.title || '新对话';
  const html = messages.length ? messages.map((message,index) => message.role === 'user' ?
    `<article class="message"><div class="message-meta"><span class="avatar user">你</span><span>你</span></div><div class="user-text">${escapeHTML(message.content)}</div></article>` : renderAnswer(message.answer || {},index)).join('') : welcome();
  $('#conversation').innerHTML = `<div class="conversation-inner">${html}${state.jobId && state.jobSessionId === state.sessionId ? '<div class="running" id="running">正在调用 Agent 与分析工具…</div>' : ''}</div>`;
  $('#conversation').querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => sendPrompt(button.dataset.prompt)));
  messages.forEach((message,index) => {
    if (message.role === 'assistant' && message.answer?.task === 'reduction') {
      message.answer.results.forEach((item,i) => drawPlot($(`#plot-${index}-${i}`), item.coordinates_sample));
    }
  });
  const conversation = $('#conversation');
  const target = conversation.querySelector('#running') || [...conversation.querySelectorAll('.message')].at(-1);
  conversation.scrollTop = target ? target.getBoundingClientRect().top - conversation.getBoundingClientRect().top + conversation.scrollTop - 12 : 0;
  $('#send').disabled = !!state.jobId;
}

async function refreshBootstrap() {
  state.bootstrap = await api('/api/bootstrap');
  state.sessions = state.bootstrap.sessions;
  renderRail(); renderInspector();
}

async function openSession(id) {
  try {
    state.sessionId = id; state.session = await api(`/api/sessions/${id}`);
    renderRail(); renderConversation();
  } catch (error) { toast(error.message); }
}

async function createSession() {
  try {
    const session = await api('/api/sessions',{method:'POST',body:'{}'});
    await refreshBootstrap(); await openSession(session.id);
    $('#prompt').focus();
  } catch (error) { toast(error.message); }
}

async function pollJob(jobId) {
  while (state.jobId === jobId) {
    await new Promise(resolve => setTimeout(resolve, 900));
    let job;
    try { job = await api(`/api/jobs/${jobId}`); }
    catch (error) { toast(error.message); state.jobId=null; renderConversation(); return; }
    if (job.state === 'running') { const running = $('#running'); if (running) running.textContent = job.stage + '…'; continue; }
    state.jobId = null; state.jobSessionId = null;
    if (job.state === 'failed') toast(job.error || '分析任务失败。');
    await refreshBootstrap(); await openSession(state.sessionId);
    return;
  }
}

async function sendPrompt(raw) {
  const content = String(raw || '').trim();
  if (!content || state.jobId) return;
  if (!state.sessionId) await createSession();
  const id = state.sessionId;
  try {
    const job = await api(`/api/sessions/${id}/messages`, {method:'POST',body:JSON.stringify({content})});
    $('#prompt').value=''; state.jobId=job.job_id; state.jobSessionId=id;
    await openSession(id); await refreshBootstrap(); pollJob(job.job_id);
  } catch (error) { toast(error.message); }
}

async function pollTrain(jobId) {
  while (state.trainId === jobId) {
    await new Promise(resolve => setTimeout(resolve, 1600));
    try {
      const job=await api(`/api/jobs/${jobId}`);
      if (job.state === 'running') { $('#train-status').textContent = job.stage || '正在训练六种分析模型。'; continue; }
      state.trainId=null;$('#train').disabled=false;
      $('#train-status').textContent=job.state === 'completed' ? '新实验已训练完成。' : `训练失败：${job.error || '查看容器日志'}`;
      await refreshBootstrap(); return;
    } catch (error) { state.trainId=null;$('#train').disabled=false;toast(error.message);return; }
  }
}

$('#new-session').addEventListener('click',createSession);
$('#composer').addEventListener('submit',event=>{event.preventDefault();sendPrompt($('#prompt').value);});
$('#prompt').addEventListener('keydown',event=>{if(event.key==='Enter' && !event.shiftKey && !event.isComposing){event.preventDefault();sendPrompt($('#prompt').value);}});
$('#train').addEventListener('click',async()=>{
  try {const job=await api('/api/train',{method:'POST',body:'{}'});state.trainId=job.job_id;$('#train').disabled=true;$('#train-status').textContent='正在训练六种分析模型，完成后更新当前实验。';pollTrain(job.job_id);}
  catch(error){toast(error.message);}
});
window.addEventListener('resize',()=>{
  for (const [index,message] of (state.session?.messages || []).entries()) if(message.role==='assistant'&&message.answer?.task==='reduction')
    message.answer.results.forEach((item,i)=>drawPlot($(`#plot-${index}-${i}`),item.coordinates_sample));
});
refreshBootstrap().then(async()=>{if(state.sessions.length) await openSession(state.sessions[0].id);else await createSession();}).catch(error=>toast(error.message));
