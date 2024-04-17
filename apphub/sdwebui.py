import os
from pathlib import Path

import gradio as gr
from apphub.app import App, AppOption
from apphub.helper import wait_for_port


class Sdwebui(App):

    @property
    def key(self):
        """key 是应用的唯一标识，用于在数据库中查找应用，所以这个值应该是唯一的"""
        return "sdwebui"

    @property
    def port(self):
        return 20000

    @property
    def op_port(self):
        return 30000

    class SdwebuiOption(AppOption):
        launch_option: str | None = ""

    cfg: SdwebuiOption

    def render_installation_page(self) -> "gr.Blocks":
        with gr.Blocks() as demo:
            gr.Markdown(
                """# 安装 Stable Diffusion Web UI

在 Featurize 中使用 Stable Diffusion Web UI 做图。

<div style="color: red">因为 SD 的安装过程比较长，因此建议将 SD 安装在云盘中，这样下次开机无需重新安装。</div>

如果将 SD 安装在云盘中，则 SD 所有的插件、模型、数据等都会被保存在云盘中，这对云盘容量的要求比较高，<strong>请随时注意云盘容量的配额<strong>，以免发生报错和产生计费。
"""
            )
            install_location = self.render_install_location(allow_work=True)
            version = gr.Dropdown(choices=["1.9.0"], label="安装的版本", value="1.9.0")
            launch_option = gr.Textbox(
                label="默认启动项", info="在启动时还可以进一步调整"
            )
            self.render_installation_button(
                inputs=[install_location, version, launch_option]
            )
            self.render_log()
        return demo

    @property
    def source_location(self):
        return os.path.join(self.cfg.install_location, "stable-diffusion-webui")

    @property
    def conda_env_name(self):
        return "sdwebui" if not self.in_work else "/cloud/app/sdwebui/env"

    @property
    def conda_mode(self):
        return "p" if self.in_work else "n"

    def render_setting_page(self):
        with gr.Blocks() as g:
            gr.Markdown("""# 设置 Stable Diffusion Web UI""")
            launch_option = gr.Textbox(
                label="默认启动项", info="在启动时还可以进一步调整"
            )
            self.render_setting_button([launch_option])
        return g

    def setting(self, launch_option):
        super().setting(launch_option)

    def installation(self, install_location, version, launch_option):
        super().installation(install_location, version, launch_option)
        self.execute_command(
            f"git clone --depth 1 --branch v{version} https://github.com/AUTOMATIC1111/stable-diffusion-webui.git",
            self.cfg.install_location,
        )
        if not self.in_work:
            self.execute_command(
                f"conda create -y -n {self.conda_env_name} python=3.10"
            )
        else:
            self.execute_command(
                f"conda create -y --prefix {self.conda_env_name} python=3.10"
            )
        with self.conda_activate(self.conda_env_name, self.conda_mode):
            self.execute_command(f"pip install 'httpx[socks]' xformers")
            source = (Path(__file__).parent / "webui-user.sh.tpl").absolute().as_posix()
            dest = os.path.join(
                self.cfg.install_location, "stable-diffusion-webui", "webui-user.sh"
            )
            self.execute_command(f"cp {source} {dest}")
            # 启动一次
            self.execute_command(
                f"bash webui.sh --listen --port {self.port}",
                daemon=True,
                cwd=self.source_location,
            )
            wait_for_port(self.port)
            self.close()

        # 调用 app_installed，标准流程，该函数会通知前端安装已经完成，切换到应用的页面
        self.app_installed()

    def start(self):
        """安装完成后，应用并不会立即开始运行，而是调用这个 start 函数。"""
        with self.conda_activate(self.conda_env_name, self.conda_mode):
            self.execute_command(
                f"bash webui.sh --listen --port {self.port} {self.cfg.launch_option}",
                daemon=True,
                cwd=self.source_location,
            )
        self.app_started()


def main():
    return Sdwebui()
