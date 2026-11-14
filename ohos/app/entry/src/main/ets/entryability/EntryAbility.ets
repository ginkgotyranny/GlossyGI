import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import entry from 'libentry.so';

// The ArkTS side is a shell: one landscape page with the XComponent the native side renders into.
// Launch arguments come as the want parameter "args" (aa start ... --ps args " --scene=bathroom"),
// one string the native side splits; launched from the home screen there are none.
export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    const args: Object | undefined = want.parameters?.['args'];
    AppStorage.setOrCreate<string>('args', args !== undefined ? args.toString() : '');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    windowStage.getMainWindow().then((win: window.Window) => {
      win.setWindowLayoutFullScreen(true).catch(() => {});
      win.setWindowKeepScreenOn(true).catch(() => {});
      win.setSpecificSystemBarEnabled('status', false).catch(() => {});
      win.setSpecificSystemBarEnabled('navigationIndicator', false).catch(() => {});
    });
    windowStage.loadContent('pages/Index');
  }

  onForeground(): void {
    entry.setForeground(true);
  }

  onBackground(): void {
    entry.setForeground(false);
  }
}
