def check_token():
    # check_token.py
    from huggingface_hub import hf_hub_download
    import os
    if token:
        try:
            print("正在尝试使用令牌下载模型配置文件...")
            # 我们只下载一个小文件来测试连接和权限
            config_path = hf_hub_download(
                repo_id="pyannote/segmentation-3.0",
                filename="config.yaml",
                use_auth_token=token
            )
            print("\n🎉 恭喜！令牌有效，模型文件下载成功！")
            print(f"配置文件已下载到: {config_path}")
            print("\n现在您可以重新运行 `uv run app.py` 了。")

        except Exception as e:
            print("\n❌ 下载失败！请检查以下问题：")
            print(f"错误信息: {e}")
            print("1. 您的令牌是否正确，并且有 'read' 权限？")
            print("2. 您是否在 Hugging Face 网站上同意了 pyannote/segmentation-3.0 的协议？")
            print("3. 您的网络代理是否已为终端开启？")


def test_api():
    import base64
    from openai import OpenAI

    client = OpenAI(base_url='http://127.0.0.1:9092/v1',api_key='123131')

    def to_data_url(path: str) -> str:
      with open(path, "rb") as fh:
        return "data:audio/wav;base64," + base64.b64encode(fh.read()).decode("utf-8")

    with open("shibie.wav", "rb") as audio_file:
      transcript = client.audio.transcriptions.create(
        model="tiny",
        file=audio_file,
        response_format="diarized_json"
      )

    print(transcript.segments)

token="" # 填写你在 huggingface.co 上具有 Read 权限的 token，去这里创建  https://huggingface.co/settings/tokens/new?tokenType=read

if __name__=="__main__":
    check_token()
    test_api()