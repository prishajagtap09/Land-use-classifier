import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import folium
from streamlit_folium import st_folium
import os
import streamlit.components.v1 as components

# ── Config ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]
CLASS_COLORS = {
    'AnnualCrop':            '#f4a261',
    'Forest':                '#2d6a4f',
    'HerbaceousVegetation':  '#74c69d',
    'Highway':               '#adb5bd',
    'Industrial':            '#e63946',
    'Pasture':               '#a8dadc',
    'PermanentCrop':         '#e9c46a',
    'Residential':           '#457b9d',
    'River':                 '#4cc9f0',
    'SeaLake':               '#023e8a',
}
CLASS_ICONS = {
    'AnnualCrop': '🌾', 'Forest': '🌲', 'HerbaceousVegetation': '🌿',
    'Highway': '🛣️', 'Industrial': '🏭', 'Pasture': '🐄',
    'PermanentCrop': '🍇', 'residential': '🏘️', 'River': '🏞️', 'SeaLake': '🌊'
}
MEAN = [0.3444, 0.3803, 0.4078]
STD  = [0.2034, 0.1367, 0.1153]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False,
                               num_classes=len(CLASS_NAMES))
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        model.eval().to(DEVICE)
        return model, True
    return model, False   # model not trained yet

model, model_ready = load_model()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ── GradCAM helper ───────────────────────────────────────────────────────────
def get_gradcam(tensor):
    gradients, activations = [], []

    def save_grad(grad):
        gradients.append(grad)

    hooks = []
    def forward_hook(m, i, o):
        activations.append(o)
        hooks.append(o.register_hook(save_grad))

    handle = model.conv_head.register_forward_hook(forward_hook)

    # Make sure tensor is 4D: (1, 3, 64, 64)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)
    out = model(tensor)
    pred = out.argmax().item()
    model.zero_grad()
    out[0, pred].backward()
    handle.remove()

    if not gradients or not activations:
        probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
        return np.zeros((8, 8)), pred, probs

    grad = gradients[0].squeeze().mean(dim=[1, 2], keepdim=True)
    act  = activations[0].squeeze()
    heatmap = (grad * act).sum(0).relu()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
    heatmap = heatmap.detach().cpu().numpy()
    probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
    return heatmap, pred, probs

def overlay_gradcam(pil_img, heatmap):
    img_arr = np.array(pil_img.resize((224, 224)))
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    ) / 255.0
    colored = (cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.55 * img_arr + 0.45 * colored).astype(np.uint8)
    return Image.fromarray(blended)

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Land Use Classifier', page_icon='🛰️', layout='wide')

st.title('🛰️ Satellite Land Use Classifier')
st.caption('EfficientNet-B0 fine-tuned on EuroSAT · 10 land cover classes')

if not model_ready:
    st.warning('⚠️ Model not found. Train the model first by running `python train_model.py`, '
               'then restart the app.')

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('About')
    st.markdown("""
**Model:** EfficientNet-B0  
**Dataset:** EuroSAT (27,000 tiles)  
**Resolution:** 64×64 px  
**Classes:** 10 land cover types
""")
    st.divider()
    st.subheader('Classes')
    for cls in CLASS_NAMES:
        color = CLASS_COLORS.get(cls, '#888')
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:2px;margin-right:6px"></span>{cls}',
            unsafe_allow_html=True
        )
    st.divider()
    st.subheader('Confidence threshold')
    threshold = st.slider('Min confidence to display', 0.0, 1.0, 0.5, 0.05)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🔍 Single image', '📦 Batch classify', '🗺️ Map view'])

# ── Tab 1: Single image ───────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap='large')

    with col1:
        st.subheader('Upload a satellite tile')
        uploaded = st.file_uploader('JPG or PNG, ideally 64×64px',
                                    type=['jpg', 'jpeg', 'png'])
        show_gradcam = st.toggle('Show GradCAM heatmap', value=True)

        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption='Uploaded tile', use_container_width=True)

    with col2:
        if uploaded and model_ready:
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            with st.spinner('Classifying...'):
                heatmap, pred_idx, probs = get_gradcam(tensor)

            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx]
            color = CLASS_COLORS.get(pred_class, '#888')

            st.subheader('Prediction')
            st.markdown(
                f'<div style="background:{color}22;border-left:4px solid {color};'
                f'padding:12px 16px;border-radius:8px;margin-bottom:12px">'
                f'<span style="font-size:24px;font-weight:600;color:{color}">'
                f'{pred_class}</span><br>'
                f'<span style="font-size:14px;color:var(--text-color,#555)">'
                f'Confidence: {confidence*100:.1f}%</span></div>',
                unsafe_allow_html=True
            )

            if show_gradcam:
                overlay = overlay_gradcam(img, heatmap)
                st.image(overlay, caption='GradCAM — regions the model focused on',
                         use_container_width=True)

            st.subheader('All class probabilities')
            sorted_idx = np.argsort(probs)[::-1]
            for i in sorted_idx:
                bar_color = CLASS_COLORS.get(CLASS_NAMES[i], '#888')
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                    f'<span style="width:130px;font-size:13px">{CLASS_NAMES[i]}</span>'
                    f'<div style="flex:1;background:#eee;border-radius:4px;height:14px">'
                    f'<div style="width:{probs[i]*100:.1f}%;background:{bar_color};'
                    f'height:14px;border-radius:4px"></div></div>'
                    f'<span style="font-size:13px;width:42px;text-align:right">'
                    f'{probs[i]*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
        elif uploaded and not model_ready:
            st.info('Train the model first, then predictions will appear here.')

# ── Tab 2: Batch classify ─────────────────────────────────────────────────────
with tab2:
    st.subheader('Classify multiple tiles at once')
    batch_files = st.file_uploader('Upload multiple satellite tiles',
                                   type=['jpg', 'jpeg', 'png'],
                                   accept_multiple_files=True)

    if batch_files and model_ready:
        results = []
        cols = st.columns(4)

        for i, f in enumerate(batch_files):
            img = Image.open(f).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out  = model(tensor)
                prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx   = prob.argmax()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = prob[pred_idx]
            color      = CLASS_COLORS.get(pred_class, '#888')

            with cols[i % 4]:
                st.image(img, use_container_width=True)
                if confidence >= threshold:
                    st.markdown(
                        f'<div style="background:{color}22;border-left:3px solid {color};'
                        f'padding:4px 8px;border-radius:4px;font-size:13px">'
                        f'<b>{pred_class}</b><br>{confidence*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.caption(f'Low confidence ({confidence*100:.1f}%)')

            results.append({'File': f.name, 'Prediction': pred_class,
                            'Confidence': f'{confidence*100:.1f}%'})

        st.divider()
        st.subheader('Summary')
        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        counts = df['Prediction'].value_counts().reset_index()
        counts.columns = ['Class', 'Count']
        st.bar_chart(counts.set_index('Class'))

# ── Tab 3: Map view ───────────────────────────────────────────────────────────
with tab3:
    st.subheader('Plot predictions on a map')
    st.caption('Upload tiles and manually enter their coordinates to visualize on a real map.')

    map_files = st.file_uploader('Upload tiles for map plotting',
                                 type=['jpg', 'jpeg', 'png'],
                                 accept_multiple_files=True,
                                 key='map_uploader')

    if map_files:
        st.info(f'{len(map_files)} tile(s) uploaded. Enter approximate coordinates below.')

        coords = []
        for i, f in enumerate(map_files):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f'**{f.name}**')
            with c2:
                lat = st.number_input(f'Lat', value=48.0 + i * 0.1,
                                      key=f'lat_{i}', format='%.4f')
            with c3:
                lon = st.number_input(f'Lon', value=11.0 + i * 0.1,
                                      key=f'lon_{i}', format='%.4f')
            coords.append((lat, lon, f))

        if st.button('Plot on map', type='primary') and model_ready:
            m = folium.Map(
                location=[coords[0][0], coords[0][1]],
                zoom_start=10,
                tiles='OpenStreetMap'   # changed from CartoDB
            )

            for lat, lon, f in coords:
                img = Image.open(f).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out  = model(tensor)
                    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
                pred_class = CLASS_NAMES[prob.argmax()]
                confidence = prob.max()
                color = CLASS_COLORS.get(pred_class, '#888888')

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=18,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(
                        f'<b>{pred_class}</b><br>'
                        f'Confidence: {confidence*100:.1f}%<br>'
                        f'Lat: {lat:.4f}, Lon: {lon:.4f}',
                        max_width=200
                    ),
                    tooltip=f'{pred_class} ({confidence*100:.0f}%)'
                ).add_to(m)

            # Save map to HTML and display it
            map_path = 'temp_map.html'
            m.save(map_path)

            with open(map_path, 'r', encoding='utf-8') as f:
                map_html = f.read()

            st.components.v1.html(map_html, height=500, scrolling=False)

            # Summary table
            st.divider()
            import pandas as pd
            results = []
            for lat, lon, f in coords:
                img = Image.open(f).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(tensor)
                    prob = torch.softmax(out, dim=1)[0].cpu().numpy()
                pred_class = CLASS_NAMES[prob.argmax()]
                confidence = prob.max()
                results.append({
                    'File': f.name,
                    'Prediction': pred_class,
                    'Confidence': f'{confidence*100:.1f}%',
                    'Lat': lat,
                    'Lon': lon
                })
            st.dataframe(pd.DataFrame(results), use_container_width=True)