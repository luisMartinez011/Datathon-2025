import streamlit as st
from predict_logic import predict_next_purchase

st.title("🔍 Predicción de Próxima Compra")

client_id = st.text_input("Ingresa el ID del cliente:")

if client_id:
    result = predict_next_purchase(client_id)

    if "error" in result:
        st.error(result["error"])
    elif not result["proxima_compra"]:
        st.warning(f"No se espera una próxima compra. (Probabilidad: {result['probabilidad']:.2f})")
    else:
        st.success("¡Se espera una próxima compra!")
        st.markdown(f"📅 *Fecha estimada:* {result['fecha_estimada']}")
        st.markdown(f"💰 *Monto estimado:* ${result['monto_estimado']}")
        st.markdown(f"🏪 *Comercio estimado:* {result['comercio_estimado']}")
        st.markdown(f"📈 *Probabilidad:* {result['probabilidad']:.2f}")
