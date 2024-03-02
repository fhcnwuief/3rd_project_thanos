import streamlit as st
import subprocess
from PIL import Image
import psutil
import time

# Set Streamlit page configuration
st.set_page_config(layout="wide")


def main():
    
    # Create an empty column on the left to center-align the button
    _, _, col1, _, _ = st.columns([1, 0.5, 3, 0.5, 1])
    _, _, col2, col3, col4, _ = st.columns([1, 1, 0.5, 1, 1, 1])

    # Add an image at the top of col1 column
    with col1:
        image = Image.open("thanos.png")  # Use a relative path if the image is in the app folder
        st.image(image, use_column_width=True)

    # Start/Stop button (centered)
    
    with col3:
        if st.button("시작"):
            script_running = True
            #st.write("마우스 제어를 시작합니다.")
            run_python_script()


# 시작 버튼을 눌렀을 때 실행되는 함수
def run_python_script():
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        try:
            global python_script_process
            # 실행할 Python 스크립트 경로를 설정합니다.
            python_script_path = "D:/THANOS/THANOS/mouse.py"  # 실행할 스크립트 파일 경로를 지정하세요.

            # 이전 메시지를 지우고 시작 메시지를 표시
            message_placeholder = st.empty()
            message_placeholder.write("마우스 제어를 시작합니다.")

            # Python 스크립트 실행 (background 실행)
            python_script_process = subprocess.Popen(["python", python_script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, encoding="utf-8")

            # # 대기하고 스크립트가 종료되면 종료 메시지 표시
            # while python_script_process.poll() is None:
            #     # Check if "filter" window is open (you may need to adapt this condition)
            #     if "filter" in psutil.Process(python_script_process.pid).open_files():
            #         message_placeholder.empty()
            #         st.write('<div style="text-align: center;">마우스 제어 중입니다.</div>', unsafe_allow_html=True)
            #     else:
            #         message_placeholder.write("마우스 제어를 시작합니다.")
            #     time.sleep(1)  # Delay for a second before checking again
            #
            # # The script has finished running
            # message_placeholder.empty()
            # st.write('<div style="text-align: center;">마우스 제어를 종료합니다.</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"오류 발생: {e}")


if __name__ == "__main__":
    main()





