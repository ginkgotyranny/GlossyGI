import { common } from '@kit.AbilityKit';
import entry from 'libentry.so';

@Entry
@Component
struct Index {
  @State status: string = 'starting';
  private timer: number = -1;

  aboutToAppear(): void {
    // Before the XComponent exists: once its surface is created the render thread starts and needs
    // the assets (resource manager), a writable folder and the launch arguments.
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    const args: string = AppStorage.get<string>('args') ?? '';
    entry.init(context.resourceManager, context.filesDir, args);
    this.timer = setInterval(() => {
      this.status = entry.getStatus();
    }, 500);
  }

  aboutToDisappear(): void {
    clearInterval(this.timer);
  }

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      // The renderer draws at the bake's resolution (16:9); the compositor scales it to this size.
      Row() {
        XComponent({ id: 'ncr', type: XComponentType.SURFACE, libraryname: 'entry' })
          .height('100%')
          .aspectRatio(16 / 9)
      }
      .width('100%')
      .height('100%')
      .justifyContent(FlexAlign.Center)

      Text(this.status)
        .fontSize(12)
        .fontColor(Color.White)
        .backgroundColor('#99000000')
        .padding(4)
        .margin(8)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }
}
