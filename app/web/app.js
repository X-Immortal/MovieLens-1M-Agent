const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const pct = (value, digits = 1) => typeof value === 'number' && Number.isFinite(value) ? `${(Number(value) * 100).toFixed(digits)}%` : '—';
const num = (value, digits = 3) => typeof value === 'number' && Number.isFinite(value) ? Number(value).toFixed(digits) : '—';
const names = {logistic_regression:'逻辑回归',decision_tree:'决策树',kmeans:'K-Means',agglomerative:'层次聚类',pca:'PCA',truncated_svd:'Truncated SVD',quality_report:'质量报告',hadoop_streaming:'Hadoop 清洗',experiment_history:'实验记录'};
const palette = ['#2d7c9a','#e1a466','#74a883','#9c83b5','#bf6f7b','#7d93bd','#a6a060','#619e9c'];
let state = {sessionId:null, session:null, sessions:[], jobId:null, jobSessionId:null, trainId:null, bootstrap:null, sending:false};

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
  $('#session-list').innerHTML = state.sessions.length ? state.sessions.map(item =>
    '<div class="session-row ' + (item.id === state.sessionId ? 'active' : '') + '"><button class="session-item" data-session="' + escapeHTML(item.id) +
    '" aria-current="' + (item.id === state.sessionId ? 'page' : 'false') + '" title="' + escapeHTML(item.title) + '">' +
    escapeHTML(item.title || '新对话') + '</button><button class="session-delete" type="button" data-delete="' + escapeHTML(item.id) +
    '" aria-label="删除对话：' + escapeHTML(item.title || '新对话') + '" title="删除对话">' +
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></svg></button></div>'
  ).join('') : '<p class="empty-history">新建对话后会保存在这里。</p>';
  $('#session-list').querySelectorAll('[data-session]').forEach(button => button.addEventListener('click', () => {
    setPanel('history', false);
    openSession(button.dataset.session);
  }));
  $('#session-list').querySelectorAll('[data-delete]').forEach(button => button.addEventListener('click', () => deleteSession(button.dataset.delete)));
}

async function deleteSession(id) {
  if (state.jobId && state.jobSessionId === id) {
    toast('此会话仍有分析任务在运行，请完成后再删除。');
    return;
  }
  const title = state.sessions.find(item => item.id === id)?.title || '新对话';
  if (!window.confirm('删除“' + title + '”？\n这段对话的消息和分析结果将被删除，无法恢复。')) return;
  try {
    await api(`/api/sessions/${id}`, {method:'DELETE'});
    if (state.sessionId === id) {
      state.sessionId = null;
      state.session = null;
      $('#prompt').value = '';
      renderConversation();
    }
    await refreshBootstrap();
    toast('对话已删除。');
    if (!state.sessionId) $('#prompt').focus();
  } catch (error) { toast(error.message); }
}

function setPanel(name, open) {
  $('#app-shell').classList.toggle(name + '-open', open);
  $('#toggle-' + name).setAttribute('aria-expanded', String(open));
  if (name === 'inspector') $('#inspector').hidden = !open;
  if (open) {
    $('#close-' + name).focus();
    if (name === 'inspector') {
      $('#app-shell').classList.remove('history-open');
      $('#toggle-history').setAttribute('aria-expanded', 'false');
    } else {
      $('#app-shell').classList.remove('inspector-open');
      $('#toggle-inspector').setAttribute('aria-expanded', 'false');
      $('#inspector').hidden = true;
    }
  }
  drawConversationPlots();
}

function fileSize(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  return bytes >= 1048576 ? (bytes / 1048576).toFixed(1) + ' MB' : Math.ceil(bytes / 1024) + ' KB';
}

function renderInspector() {
  const b = state.bootstrap;
  if (!b) return;
  const notices = [];
  if (!b.deepseek_configured) notices.push('请配置 DeepSeek API 后再开始分析。');
  if (!b.model_ready) notices.push(b.quality ? '模型尚未就绪，可在“分析资料”的实验管理中训练。' : '请先执行数据清洗，再训练模型。');
  $('#setup-notice').textContent = notices.join(' ');
  $('#setup-notice').hidden = !notices.length;
  $('#data-status').innerHTML =
    '<div class="data-line"><span>清洗数据</span><strong>' + (b.quality ? '已生成' : '待清洗') + '</strong></div>' +
    '<div class="data-line"><span>分析模型</span><strong>' + (b.model_ready ? '已就绪' : '待训练') + '</strong></div>' +
    '<div class="data-line"><span>DeepSeek API</span><strong>' + (b.deepseek_configured ? '已配置' : '未配置') + '</strong></div>' +
    '<dl class="data-version">' +
    (b.quality?.split_counts ? '<dt>训练 / 验证 / 测试</dt><dd>' + Object.values(b.quality.split_counts).map(value => Number(value).toLocaleString()).join(' / ') + '</dd>' : '') +
    (b.quality?.boundaries?.T1 ? '<dt>时间边界</dt><dd>T1 ' + escapeHTML(b.quality.boundaries.T1) + '<br>T2 ' + escapeHTML(b.quality.boundaries.T2) + '</dd>' : '') +
    (b.experiment ? '<dt>模型版本</dt><dd>' + escapeHTML(b.experiment.experiment_id) + '</dd>' : '') +
    (b.quality ? '<dt>数据版本</dt><dd>' + escapeHTML(b.quality.data_version) + '</dd>' : '') + '</dl>';
  const files = b.artifacts || [];
  const links = entries => entries.map(file =>
    '<a class="artifact-link" href="/api/artifacts/' + encodeURIComponent(file.id) + '" download><span class="file-label">' +
    escapeHTML(file.label) + '</span><span class="size">' + fileSize(file.size) + '</span></a>'
  ).join('') || '<p class="side-copy">完成清洗和训练后可下载。</p>';
  $('#report-list').innerHTML = links(files.filter(file => file.filename.endsWith('.json')));
  $('#artifact-list').innerHTML = links(files.filter(file => !file.filename.endsWith('.json')));
}

function welcome() {
  const examples = [
    '用户 1 会给电影 1 打高分吗？',
    '比较 K-Means 与层次聚类的用户群体。',
    '比较 PCA 与 SVD 的二维降维结果。',
    '清洗前后的数据质量有什么变化？'
  ];
  return '<section class="welcome"><h2>想了解电影评分的哪一面？</h2>' +
    '<p>查询评分预测、用户偏好或数据质量，也可以围绕结果继续追问。</p>' +
    '<div class="prompt-list">' + examples.map(prompt =>
      '<button type="button" class="suggestion" data-prompt="' + escapeHTML(prompt) + '">' + escapeHTML(prompt) + '</button>'
    ).join('') + '</div></section>';
}

function table(headers, rows, label) {
  return '<div class="table-scroll" role="region" aria-label="' + escapeHTML(label) + '" tabindex="0">' +
    '<table class="data-table"><thead><tr>' + headers.map(value => '<th scope="col">' + escapeHTML(value) + '</th>').join('') +
    '</tr></thead><tbody>' + rows.map(row => '<tr>' + row.map(value => '<td>' + escapeHTML(value ?? '—') + '</td>').join('') + '</tr>').join('') +
    '</tbody></table></div>';
}

function prose(text) {
  const blocks = [];
  let paragraph = [], list = [];
  const flushParagraph = () => {
    if (paragraph.length) blocks.push('<p>' + paragraph.map(escapeHTML).join('<br>') + '</p>');
    paragraph = [];
  };
  const flushList = () => {
    if (list.length) blocks.push('<ul>' + list.map(line => '<li>' + escapeHTML(line) + '</li>').join('') + '</ul>');
    list = [];
  };
  for (const line of String(text || '').trim().split('\n')) {
    if (!line.trim()) { flushParagraph(); flushList(); }
    else if (/^\s*[-•]\s+/.test(line)) { flushParagraph(); list.push(line.replace(/^\s*[-•]\s+/, '')); }
    else { flushList(); paragraph.push(line); }
  }
  flushParagraph(); flushList();
  return blocks.join('');
}

function detail(data, label = '原始结果') {
  return '<details class="details"><summary>' + escapeHTML(label) + '</summary><pre>' + escapeHTML(JSON.stringify(data, null, 2)) + '</pre></details>';
}

function renderQuality(report) {
  const quality = report.quality || {};
  if (!quality.before || !quality.after) return detail(report);
  const labels = {Accurate:'准确性',Complete:'完整性',Unique:'唯一性','Up-to-date':'时效性',Consistent:'一致性'};
  const tables = {users:'用户',movies:'电影',ratings:'评分'};
  return '<h3>清洗前后质量评分</h3>' + table(['维度','清洗前','清洗后','变化'], Object.keys(quality.before).map(key => {
    const change = quality.after[key] - quality.before[key];
    return [labels[key] || key, num(quality.before[key],2), num(quality.after[key],2), (change > 0 ? '+' : '') + num(change,2)];
  }), '清洗前后质量评分') + '<h3>数据处置</h3>' +
    table(['数据','清洗前','清洗后','修复','去重','冲突隔离','无效隔离'], Object.entries(report.cleaning || {}).map(([name,row]) => [
      tables[name] || name, quality.summaries?.before?.[name]?.total, row.cleaned, row.repaired, row.duplicates, row.conflicts, row.rejected
    ]), '清洗数据量与处置') + '<p class="small-note">修复计数属于保留记录；去重和隔离表示移除记录，隔离数据未被修复。</p>';
}

function qualityEvidence(report) {
  const tableNames = {users:'用户',movies:'电影',ratings:'评分'};
  const reasons = {field_count_or_missing_value:'字段数量错误或缺失',occupation_or_zip_invalid:'职业或邮编格式无效',
    user_demographic_invalid:'用户属性类别无效',unknown_genre:'未知电影类型',non_numeric_field:'数值字段无法解析',
    rating_out_of_range_or_id_invalid:'评分超出范围或编号无效',timestamp_out_of_expected_range:'时间戳超出历史范围',
    unknown_movie:'电影引用不存在',unknown_user:'用户引用不存在'};
  const examples = Object.entries(report.cleaning || {}).flatMap(([name,row]) => [
    ...Object.entries(row.reject_examples || {}).flatMap(([reason,lines]) => lines.slice(0,1).map(line => [tableNames[name], '隔离', reasons[reason] || reason, line])),
    ...(row.duplicate_examples || []).slice(0,1).map(line => [tableNames[name], '去重', '业务键与内容重复', line]),
    ...(row.conflict_examples || []).slice(0,1).map(line => {
      let raw = line;
      try { raw = JSON.parse(line).raw || line; } catch {}
      return [tableNames[name], '冲突隔离', '同一业务键内容冲突', raw];
    })
  ]);
  let html = '<h4>评分方法</h4>' + table(['维度','口径'], Object.entries(report.quality?.method || {}), '质量评分口径') +
    '<h4>异常记录与处理原因</h4>' + table(['数据','处置','原因','原始记录'], examples, '异常记录样例');
  const headers = {users:['用户','性别','年龄段','职业','邮编'],movies:['电影','标题','类型'],ratings:['用户','电影','评分','时间戳']};
  for (const [name,rows] of Object.entries(report.clean_samples || {})) {
    html += '<h4>清洗后' + escapeHTML(tableNames[name] || name) + '样例</h4>' +
      table(headers[name] || ['记录'], rows.map(line => line.split('::')), '清洗后' + (tableNames[name] || name) + '样例');
  }
  return html;
}

function classificationMetrics(results) {
  return table(['模型','Precision','Recall','F1','ROC-AUC'], results.map(item => {
    const metrics = item.test_metrics || item.test || {};
    return [names[item.tool] || item.tool, num(metrics.precision), num(metrics.recall), num(metrics.f1), num(metrics.roc_auc)];
  }), '分类模型测试集指标');
}

function renderClassification(results) {
  const first = results[0];
  if (first.positive_probability === undefined) return '<h3>分类模型测试集指标</h3>' + classificationMetrics(results);
  const warnings = [...new Set(results.map(item => item.history_warning).filter(Boolean))];
  return '<h3>高分预测</h3><div class="result-context"><strong>' + escapeHTML(first.movie_title) +
    '</strong><span>用户 ' + escapeHTML(first.UserID) + ' / 电影 ' + escapeHTML(first.MovieID) + '</span></div>' +
    table(['模型','高分概率','阈值','预测结果'], results.map(item => [
      names[item.tool] || item.tool, pct(item.positive_probability), pct(item.threshold), item.predicted_high_rating ? '高分' : '非高分'
    ]), '高分预测对照') +
    '<p class="small-note">高分指评分 ≥ 4。' + (first.target_pair_in_training ? '该组合曾在训练期出现，本次估计已扣除其历史评分。' : '') + '</p>' +
    warnings.map(text => '<p class="warning-note">' + escapeHTML(text) + '</p>').join('');
}

function renderClustering(results) {
  const first = results[0];
  let html = '<h3>用户群体</h3>' + table(
    ['算法','群体数','轮廓系数', ...(first.UserID ? ['用户 ' + first.UserID + ' 的群体'] : [])],
    results.map(item => [names[item.tool] || item.tool, item.n_clusters, num(item.silhouette,5), ...(first.UserID ? [item.cluster] : [])]),
    '聚类算法对照');
  if (first.selected_users) {
    html += table(['用户', ...results.map(item => names[item.tool] || item.tool)], first.selected_users.map(row => [
      row.UserID, ...results.map(item => item.selected_users?.find(user => user.UserID === row.UserID)?.cluster)
    ]), '指定用户的群体归属');
  }
  html += '<p class="small-note">基于 ' + Number(first.users).toLocaleString() + ' 名用户的训练期偏好画像。</p>';
  return html + [...new Set(results.map(item => item.history_warning).filter(Boolean))].map(text =>
    '<p class="warning-note">' + escapeHTML(text) + '</p>').join('');
}

function renderReduction(results, suffix) {
  const first = results[0];
  let html = '<h3>降维结果</h3><p class="small-note">' + escapeHTML(first.original_dimensions) + ' 维压缩到 ' +
    escapeHTML(first.components) + ' 维，覆盖 ' + Number(first.coordinate_count).toLocaleString() + ' 名用户。</p>' +
    table(['方法','方差保留','重构误差'], results.map(item => [
      names[item.tool] || item.tool, pct(item.explained_variance_ratio_sum), num(item.reconstruction_mse,5)
    ]), '降维效果对照');
  for (const item of results.filter(row => row.retention_threshold !== undefined)) {
    html += '<p class="' + (item.retention_met ? 'small-note' : 'warning-note') + '">' +
      escapeHTML(names[item.tool] || item.tool) + (item.retention_met ? '达到' : '未达到') + '要求的 ' +
      pct(item.retention_threshold) + ' 方差保留比例。</p>';
  }
  html += '<div class="plot-grid">' + results.map((item,index) => {
    const groups = [...new Set((item.coordinates_sample || []).map(row => Number(row.kmeans)))].sort((a,b) => a-b);
    const label = names[item.tool] || item.tool;
    return '<div class="plot-box"><h4>' + escapeHTML(label) + '</h4><div class="plot-sub">' +
      Number((item.coordinates_sample || []).length).toLocaleString() + ' 个用户坐标</div>' +
      '<canvas id="plot-' + suffix + '-' + index + '" aria-label="' + escapeHTML(label) + ' 用户前两个分量的散点图"></canvas>' +
      '<div class="legend">' + groups.map(group => '<span><i style="background:' + palette[group % palette.length] +
        '" aria-hidden="true"></i>群体 ' + group + '</span>').join('') + '</div>' +
      '<button type="button" class="text-button plot-download" data-plot="plot-' + suffix + '-' + index +
      '" data-label="' + escapeHTML(label) + '" data-version="' + escapeHTML(item.experiment_id) +
      '" data-groups="' + groups.join(',') + '">下载 ' + escapeHTML(label) + ' 图</button></div>';
  }).join('') + '</div><p class="small-note">图中颜色对应原始画像空间的 K-Means 群体。</p>';
  if (results.some(item => item.selected_users)) {
    html += table(['方法','用户','坐标 1','坐标 2'], results.flatMap(item => (item.selected_users || []).map(row => [
      names[item.tool] || item.tool, row.UserID, num(row.component_1), num(row.component_2)
    ])), '指定用户的降维坐标');
  }
  return html;
}

function renderHistory(results) {
  return '<h3>实验记录</h3>' + table(['实验','维数','逻辑回归 F1','K-Means 轮廓系数'], results.map(item => [
    item.experiment_id, item.components, num(item.logistic_test_f1), num(item.kmeans_silhouette,5)
  ]), '已登记实验');
}

function resultGroups(answer) {
  const groups = new Map();
  for (const item of answer.results || []) {
    const task = item.task || answer.task;
    const selected = item.selected_users?.map(row => row.UserID) || (item.UserID ? [item.UserID] : []);
    const key = [task, JSON.stringify(selected), item.MovieID || '', item.components || ''].join(':');
    if (!groups.has(key)) groups.set(key, {task, results:[]});
    groups.get(key).results.push(item);
  }
  return [...groups.values()];
}

function renderEvidence(answer) {
  const sections = resultGroups(answer).map(({task, results}) => {
    if (task === 'classification') {
      return (results[0].positive_probability !== undefined ? '<h4>测试集指标</h4>' + classificationMetrics(results) : '') +
        table(['模型','参数'], results.map(item => [names[item.tool] || item.tool, item.parameter]), '分类模型参数') +
        '<p>' + escapeHTML(results[0].history_policy || '评价使用独立测试时间段。') + '</p>';
    }
    if (task === 'clustering') {
      return results.map(item => '<h4>' + escapeHTML(names[item.tool] || item.tool) + ' 群体画像</h4>' +
        table(['群体','人数','平均评分','相对偏好','代表用户'], (item.groups || []).map(row => [
          row.cluster, row.users, num(row.mean_rating,2), row.explanation, (row.representative_user_ids || []).join('、')
        ]), (names[item.tool] || item.tool) + ' 群体画像')
      ).join('') + '<p>群体偏好来自训练期评分统计，用户集合查询使用已有模型。</p>' +
      (results[0].profiles_sample ? '<h4>训练期画像样例（最多 5 人）</h4>' +
        table(['用户','评分数','平均评分','主要类型偏差'], results[0].profiles_sample.map(row => [
          row.UserID, row.rating_count, num(row.overall_mean,2),
          Object.entries(row).filter(([key]) => key.startsWith('preference:')).sort((a,b) => b[1]-a[1]).slice(0,3)
            .map(([key,value]) => key.slice(11) + ' ' + num(value,2)).join('；')
        ]), '用户画像样例') + '<p>类型偏差相对于该用户自身均分计算，并使用 5 条评分平滑。</p>' : '');
    }
    if (task === 'reduction') {
      return '<h4>主要变化方向</h4>' + table(['方法','分量','方差比例','主要特征与权重'], results.flatMap(item =>
        (item.component_directions || []).map(row => [names[item.tool] || item.tool, row.component, pct(row.explained_variance_ratio),
          row.top_features.map(feature => feature.feature + ' ' + num(feature.weight)).join('；')
        ])), '主要分量特征权重') + '<p>' + escapeHTML(results[0].direction_note || '当前历史结果未保存分量权重，可重新查询。') + '</p>' +
        '<h4>降维前后聚类对照</h4>' + table(['方法','聚类','原空间轮廓系数','降维后轮廓系数','原空间耗时','降维后耗时'], results.flatMap(item =>
        Object.entries(item.comparison || {}).map(([name,row]) => [
          names[item.tool] || item.tool, (names[name] || name) + ' / ' + item.comparison_groups?.[name] + ' 群',
          num(row.original_silhouette,5), num(row.reduced_silhouette,5), num(row.original_cluster_seconds) + ' 秒', num(row.reduced_cluster_seconds) + ' 秒'
        ])), '降维前后聚类对照') +
        '<p>两组轮廓系数按各自空间的距离计算。方差保留比例用于衡量降维效果。</p>' +
        '<h4>资源用量</h4>' + table(['方法','变换耗时','输入矩阵','坐标数组'], results.map(item => [
          names[item.tool] || item.tool, num(item.transform_seconds,3) + ' 秒', fileSize(item.input_matrix_bytes), fileSize(item.coordinate_bytes)
        ]), '降维资源用量') + '<p>' + escapeHTML(results[0].resource_note) + '</p>';
    }
    if (task === 'quality') return results.map(qualityEvidence).join('');
    return '';
  }).join('');
  const calls = answer.tool_calls?.length ? '<h4>工具调用</h4><pre>' + escapeHTML(JSON.stringify(answer.tool_calls, null, 2)) + '</pre>' : '';
  const version = answer.experiment_id || answer.data_version ? '<h4>结果版本</h4><p>' +
    [answer.experiment_id, answer.data_version].filter(Boolean).map(escapeHTML).join('<br>') + '</p>' : '';
  const parameters = (answer.results || []).filter(item => item.estimator_params);
  const configuration = parameters.length ? '<h4>实际模型参数</h4><pre>' +
    escapeHTML(JSON.stringify(parameters.map(item => ({algorithm:item.tool, parameters:item.estimator_params})), null, 2)) + '</pre>' : '';
  const contexts = [...new Map((answer.results || []).filter(item => item.boundaries).map(item => [
    [item.data_version, item.rule_version, item.boundaries.T1, item.boundaries.T2].join(':'), item
  ])).values()];
  const splits = contexts.map(item => '<h4>时间切分与规则</h4><p>规则 ' + escapeHTML(item.rule_version) + '<br>T1 ' +
    escapeHTML(item.boundaries.T1) + '<br>T2 ' + escapeHTML(item.boundaries.T2) + '</p>' +
    table(['训练评分','验证评分','测试评分'], [[item.split_counts?.train ?? item.time_split?.after?.train,
      item.split_counts?.validation ?? item.time_split?.after?.validation, item.split_counts?.test ?? item.time_split?.after?.test]], '结果使用的数据切分')).join('');
  return sections || calls || version ? '<details class="details"><summary>分析依据</summary><div class="evidence">' +
    sections + configuration + splits + calls + version + '</div></details>' : '';
}

function renderAnswer(answer, index) {
  const status = answer.status === 'partial' ? '部分完成' : answer.status === 'failed' ? '分析未完成' : '';
  const body = resultGroups(answer).map(({task, results}, groupIndex) => {
    let content = '';
    if (task === 'quality') content = results.map(renderQuality).join('');
    else if (task === 'classification') content = renderClassification(results);
    else if (task === 'clustering') content = renderClustering(results);
    else if (task === 'reduction') content = renderReduction(results, index + '-' + groupIndex);
    else if (task === 'history') content = renderHistory(results.flatMap(item => item.experiments || [item]));
    else content = detail(results);
    return '<section class="result-section">' + content + '</section>';
  }).join('');
  return '<article class="message"><div class="message-meta"><span>MovieLens Agent</span>' +
    (status ? '<span class="answer-state">' + status + '</span>' : '') + '</div><div class="answer">' +
    (answer.error ? '<p class="error-note">' + escapeHTML(answer.error) + '</p>' : '') +
    '<div class="answer-prose">' + prose(answer.summary) + prose(answer.explanation) + '</div>' +
    body + renderEvidence(answer) + '</div></article>';
}

function drawAnswerPlots(answer, index) {
  resultGroups(answer).forEach(({task, results}, groupIndex) => {
    if (task === 'reduction') results.forEach((item, i) =>
      drawPlot($('#plot-' + index + '-' + groupIndex + '-' + i), item.coordinates_sample));
  });
}

function drawConversationPlots() {
  for (const [index,message] of (state.session?.messages || []).entries()) {
    if (message.role === 'assistant') drawAnswerPlots(message.answer || {}, index);
  }
}

function drawPlot(canvas, rows) {
  if (!canvas || !rows?.length) return;
  const width = canvas.clientWidth, height = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
  if (!width || !height) return;
  canvas.width = Math.round(width*dpr); canvas.height = Math.round(height*dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const values = rows.filter(row => Number.isFinite(Number(row.component_1)) && Number.isFinite(Number(row.component_2)));
  if (!values.length) return;
  const xs = values.map(x=>Number(x.component_1)), ys = values.map(x=>Number(x.component_2));
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const left=44, right=12, top=20, bottom=40, w=width-left-right,h=height-top-bottom;
  ctx.fillStyle='#596d7e';ctx.font='11px sans-serif';
  for (let tick=0; tick<=2; tick++) {
    const x=left+w*tick/2, y=top+h-h*tick/2;
    ctx.textAlign='center';ctx.fillText((xmin+(xmax-xmin)*tick/2).toFixed(1),x,height-bottom+16);
    ctx.textAlign='right';ctx.fillText((ymin+(ymax-ymin)*tick/2).toFixed(1),left-6,y+4);
  }
  ctx.textAlign='center';ctx.fillText('分量 1',left+w/2,height-5);
  ctx.textAlign='left';ctx.fillText('分量 2',left,12);
  ctx.strokeStyle='#d9e4eb';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,height-bottom);ctx.lineTo(width-right,height-bottom);ctx.moveTo(left,top);ctx.lineTo(left,height-bottom);ctx.stroke();
  for (const row of values) {
    const x=left+(Number(row.component_1)-xmin)/((xmax-xmin)||1)*w;
    const y=height-bottom-(Number(row.component_2)-ymin)/((ymax-ymin)||1)*h;
    ctx.fillStyle=palette[Number(row.kmeans)%palette.length] || palette[0];ctx.globalAlpha=.64;
    ctx.beginPath();ctx.arc(x,y,2.1,0,Math.PI*2);ctx.fill();
  }
  ctx.globalAlpha=1;
}

function downloadPlot(button) {
  const source = document.getElementById(button.dataset.plot);
  if (!source?.width) return;
  const canvas = document.createElement('canvas');
  const scale = source.width / source.clientWidth;
  const width = Math.max(source.clientWidth, 440), height = source.clientHeight + 112;
  canvas.width = width * scale; canvas.height = height * scale;
  const ctx = canvas.getContext('2d'); ctx.scale(scale, scale);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,width,height);
  ctx.fillStyle = '#233c50'; ctx.font = 'bold 16px sans-serif';
  ctx.fillText(button.dataset.label + ' 用户投影',16,24);
  ctx.drawImage(source, (width-source.clientWidth)/2, 34, source.clientWidth, source.clientHeight);
  ctx.font = '12px sans-serif';
  const groups = button.dataset.groups.split(',').filter(Boolean).map(Number);
  groups.forEach((group,index) => {
    ctx.fillStyle = palette[group % palette.length]; ctx.fillRect(16+index*70,height-60,8,8);
    ctx.fillStyle = '#233c50'; ctx.fillText('群体 '+group,29+index*70,height-52);
  });
  ctx.font = '11px sans-serif'; ctx.fillStyle = '#596d7e';
  ctx.fillText('颜色：原空间 K-Means 群体；展示坐标样本',16,height-32);
  ctx.fillText('实验：'+button.dataset.version,16,height-13);
  canvas.toBlob(blob => {
    if (!blob) return;
    const link=document.createElement('a'), url=URL.createObjectURL(blob);
    link.href=url; link.download=button.dataset.label.replace(/\s+/g,'-')+'-'+button.dataset.version+'.png';
    link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }, 'image/png');
}

function renderConversation() {
  const messages = state.session?.messages || [];
  $('#conversation-title').textContent = state.session?.title || '新对话';
  const html = messages.length ? messages.map((message,index) => message.role === 'user' ?
    `<article class="message"><div class="message-meta">你</div><div class="user-text">${escapeHTML(message.content)}</div></article>` : renderAnswer(message.answer || {},index)).join('') : welcome();
  $('#conversation').innerHTML = `<div class="conversation-inner">${html}${state.jobId && state.jobSessionId === state.sessionId ? '<div class="running" id="running" role="status">正在分析…</div>' : ''}</div>`;
  $('#conversation').querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => sendPrompt(button.dataset.prompt)));
  drawConversationPlots();
  $('#conversation').setAttribute('aria-busy', String(Boolean(state.jobId && state.jobSessionId === state.sessionId)));
  const conversation = $('#conversation');
  const target = conversation.querySelector('#running') || [...conversation.querySelectorAll('.message')].at(-1);
  conversation.scrollTop = target ? target.getBoundingClientRect().top - conversation.getBoundingClientRect().top + conversation.scrollTop - 12 : 0;
  $('#send').disabled = !!state.jobId || state.sending;
}

async function refreshBootstrap() {
  state.bootstrap = await api('/api/bootstrap');
  state.sessions = state.bootstrap.sessions;
  renderRail(); renderInspector();
}

async function openSession(id) {
  try {
    state.sessionId = id;
    const session = await api(`/api/sessions/${id}`);
    if (state.sessionId !== id) return;
    state.session = session;
    renderRail(); renderConversation();
  } catch (error) { toast(error.message); }
}

function createSession() {
  state.sessionId = null; state.session = null;
  $('#prompt').value = '';
  renderRail(); renderConversation();
  setPanel('history', false);
  $('#prompt').focus();
}

async function pollJob(jobId) {
  while (state.jobId === jobId) {
    await new Promise(resolve => setTimeout(resolve, 900));
    let job;
    try { job = await api(`/api/jobs/${jobId}`); }
    catch (error) { toast(error.message); state.jobId=null; renderConversation(); return; }
    if (job.state === 'running') {
      const running = $('#running');
      const tool = job.stage?.startsWith('正在执行 ') ? job.stage.slice(5) : null;
      if (running) running.textContent = tool && names[tool] ? `正在执行${names[tool]}…` : job.stage || '正在分析…';
      continue;
    }
    state.jobId = null; state.jobSessionId = null;
    if (job.state === 'failed') toast(job.error || '分析任务失败。');
    $('#response-status').textContent = job.state === 'failed' ? '分析任务失败。' : '回答已生成。';
    await refreshBootstrap();
    if (state.sessionId) await openSession(state.sessionId);
    else renderConversation();
    return;
  }
}

async function sendPrompt(raw) {
  const content = String(raw || '').trim();
  if (!content || state.jobId || state.sending) return;
  state.sending = true;
  $('#send').disabled = true;
  $('#response-status').textContent = '';
  try {
    if (!state.sessionId) {
      state.session = await api('/api/sessions', {method:'POST',body:'{}'});
      state.sessionId = state.session.id;
    }
    const id = state.sessionId;
    const job = await api(`/api/sessions/${id}/messages`, {method:'POST',body:JSON.stringify({content})});
    $('#prompt').value=''; state.jobId=job.job_id; state.jobSessionId=id;
    await openSession(id); await refreshBootstrap(); pollJob(job.job_id);
  } catch (error) { toast(error.message); }
  finally { state.sending = false; $('#send').disabled = !!state.jobId; }
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
$('#conversation').addEventListener('click',event=>{
  const button=event.target.closest('.plot-download');
  if(button) downloadPlot(button);
});
$('#composer').addEventListener('submit',event=>{event.preventDefault();sendPrompt($('#prompt').value);});
$('#prompt').addEventListener('keydown',event=>{if(event.key==='Enter' && !event.shiftKey && !event.isComposing){event.preventDefault();sendPrompt($('#prompt').value);}});
$('#train').addEventListener('click',async()=>{
  try {const job=await api('/api/train',{method:'POST',body:'{}'});state.trainId=job.job_id;$('#train').disabled=true;$('#train-status').textContent='正在训练六种分析模型，完成后更新当前实验。';pollTrain(job.job_id);}
  catch(error){toast(error.message);}
});
window.addEventListener('resize', drawConversationPlots);
for (const panel of ['history', 'inspector']) {
  $('#toggle-' + panel).addEventListener('click', () => setPanel(panel, $('#toggle-' + panel).getAttribute('aria-expanded') !== 'true'));
  $('#close-' + panel).addEventListener('click', () => {
    setPanel(panel, false);
    $('#toggle-' + panel).focus();
  });
}
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  for (const panel of ['history', 'inspector']) {
    if ($('#toggle-' + panel).getAttribute('aria-expanded') === 'true') {
      setPanel(panel, false);
      $('#toggle-' + panel).focus();
    }
  }
});
refreshBootstrap().then(async()=>{if(state.sessions.length) await openSession(state.sessions[0].id);else createSession();}).catch(error=>toast(error.message));
