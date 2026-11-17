import { common } from '@kit.AbilityKit';
import entry from 'libentry.so';

// The page: the XComponent the native side renders into, and over it the panel, which is what the
// desktop shows in Mogwai for this project: the scene and its animation (run.bat's scene, variant
// and camera options; Mogwai's time controls), the NeuralRadiosity pass's own panel
// (NeuralRadiosity::renderUI) and frame times. ArkUI draws and updates the panel on its own, so it
// stays responsive whatever the renderer's frame time is.
//
// The native side publishes its state as JSON (entry.getState), polled here; a control calls
// entry.set(name, value) or entry.command(name, argument), which the render thread applies at the
// start of its next frame. Touches on the picture go to the native side, which moves the camera
// (dispatchTouchEvent in app.cpp).

interface SceneInfo {
  name: string;
  variants: string[];
}

interface Caps {
  animated: boolean;
  postFilter: boolean;
  lights: boolean;
  renderPath: boolean;
  train: boolean;
  savedModel: boolean;
}

interface ScheduleState {
  heat: number;
  holdLeft: number;
  learningRate: number;
  emaDecay: number;
}

interface AnimState {
  time: number;
  length: number;
  sample: number;
  samples: number;
}

interface InfoState {
  width: number;
  height: number;
  batchSize: number;
  numRHS: number;
  specRays: number;
  networks: string;
  model: string;
}

interface Timing {
  name: string;
  ms: number;
}

interface AppState {
  status: string;
  message?: string;
  scenes?: SceneInfo[];
  scene?: string;
  variant?: string;
  loaded?: boolean;
  frame?: number;
  frameMs?: number;
  presentMs?: number;
  smoothMs?: number;
  trainSteps?: number;
  events?: number;
  caps?: Caps;
  settings?: Record<string, number>;
  schedule?: ScheduleState;
  anim?: AnimState;
  info?: InfoState;
  timings?: Timing[];
}

// Arguments of the @Builder functions that show changing text. ArkUI updates what a builder built only
// when its one argument is an object literal (passed by reference); a builder's plain arguments are
// copied once, so a line built from them would keep its first text.
class LineArgs {
  text: string = '';
}

class HeaderArgs {
  title: string = '';
  section: string = '';
}

const MODES: string[] = ['Idle', 'Render', 'Train', 'Online Train'];   // NeuralRadiosity::RenderMode
const ENV_ENCODINGS: string[] = ['Linear', 'Signed log', 'Log'];
const TEXT: string = '#FFFFFFFF';
const DIM: string = '#FFB0B0B0';
const ACCENT: string = '#FF3D7EFF';
const IDLE: string = '#FF4A4A4A';

@Entry
@Component
struct Index {
  @State state: AppState = { status: 'starting' };
  @State showPanel: boolean = true;
  @State showScene: boolean = true;
  @State showPass: boolean = true;
  @State showSchedule: boolean = false;
  @State showStats: boolean = false;
  @State tick: number = 0;   // bumped to redraw at once after a control was used
  private timer: number = -1;
  // What a control was just set to: shown until the native side reports it (or a moment has passed).
  private pending: Map<string, number> = new Map<string, number>();
  private pendingSince: Map<string, number> = new Map<string, number>();

  aboutToAppear(): void {
    // Before the XComponent exists: once its surface is created the render thread starts and needs
    // the assets (resource manager), a writable folder and the launch arguments.
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    const args: string = AppStorage.get<string>('args') ?? '';
    entry.init(context.resourceManager, context.filesDir, args);
    this.timer = setInterval(() => {
      this.poll();
    }, 250);
  }

  aboutToDisappear(): void {
    clearInterval(this.timer);
  }

  poll(): void {
    try {
      this.state = JSON.parse(entry.getState()) as AppState;
    } catch (e) {
      return;
    }
    const now: number = Date.now();
    const names: string[] = Array.from(this.pending.keys());
    for (let i = 0; i < names.length; i++) {
      const since: number = this.pendingSince.get(names[i]) ?? 0;
      if (now - since > 1500 || this.reported(names[i]) === this.pending.get(names[i])) {
        this.pending.delete(names[i]);
        this.pendingSince.delete(names[i]);
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Values.
  reported(name: string): number {
    const settings: Record<string, number> | undefined = this.state.settings;
    if (settings === undefined) {
      return 0;
    }
    const value: number | undefined = settings[name];
    return value === undefined ? 0 : value;
  }

  value(name: string): number {
    if (this.tick < 0) {
      return 0;   // never: reading the counter makes what shows a value redraw when a control was used
    }
    const pending: number | undefined = this.pending.get(name);
    return pending === undefined ? this.reported(name) : pending;
  }

  setValue(name: string, value: number): void {
    this.pending.set(name, value);
    this.pendingSince.set(name, Date.now());
    entry.set(name, value);
    this.tick++;
  }

  isOpen(section: string): boolean {
    if (section === 'scene') {
      return this.showScene;
    }
    if (section === 'pass') {
      return this.showPass;
    }
    if (section === 'schedule') {
      return this.showSchedule;
    }
    return this.showStats;
  }

  toggleSection(section: string): void {
    if (section === 'scene') {
      this.showScene = !this.showScene;
    } else if (section === 'pass') {
      this.showPass = !this.showPass;
    } else if (section === 'schedule') {
      this.showSchedule = !this.showSchedule;
    } else {
      this.showStats = !this.showStats;
    }
  }

  loaded(): boolean {
    return this.state.loaded === true;
  }

  cap(name: string): boolean {
    const caps: Caps | undefined = this.state.caps;
    if (caps === undefined) {
      return false;
    }
    if (name === 'animated') {
      return caps.animated;
    }
    if (name === 'postFilter') {
      return caps.postFilter;
    }
    if (name === 'lights') {
      return caps.lights;
    }
    if (name === 'fxaa') {
      return caps.postFilter && this.value('postFilter') !== 0;
    }
    if (name === 'savedModel') {
      return caps.savedModel;
    }
    if (name === 'train') {
      return caps.train;
    }
    return true;
  }

  variantsOf(scene: string): string[] {
    const scenes: SceneInfo[] = this.state.scenes ?? [];
    for (let i = 0; i < scenes.length; i++) {
      if (scenes[i].name === scene) {
        return scenes[i].variants;
      }
    }
    return [];
  }

  pickScene(scene: string, variant: string): void {
    const variants: string[] = this.variantsOf(scene);
    if (variants.length === 0) {
      return;
    }
    const chosen: string = variants.indexOf(variant) >= 0 ? variant : variants[0];
    entry.command('scene', scene + '/' + chosen);
  }

  sceneText(): string {
    const scene: string = this.state.scene ?? '';
    return scene.length > 0 ? scene + ' (' + (this.state.variant ?? '') + ')' : 'no scene';
  }

  frameText(): string {
    const ms: number = this.state.smoothMs ?? 0;
    return ms > 0 ? ms.toFixed(0) + ' ms  ' + (1000 / ms).toFixed(1) + ' fps' : '';
  }

  eventsText(): string {
    const events: number = this.state.events ?? 0;
    const names: string[] = [];
    if ((events & 1) !== 0) {
      names.push('camera');
    }
    if ((events & 2) !== 0) {
      names.push('edit');
    }
    if ((events & 4) !== 0) {
      names.push('light');
    }
    if ((events & 8) !== 0) {
      names.push('object');
    }
    return names.length > 0 ? names.join('+') : 'none';
  }

  scheduleText(): string {
    const s: ScheduleState | undefined = this.state.schedule;
    if (s === undefined) {
      return '';
    }
    return 'Heat ' + s.heat.toFixed(2) + '  hold ' + s.holdLeft.toString();
  }

  rateText(): string {
    const s: ScheduleState | undefined = this.state.schedule;
    if (s === undefined) {
      return '';
    }
    return 'LR ' + s.learningRate.toExponential(2) + '  EMA ' + s.emaDecay.toFixed(4) + '  events ' + this.eventsText();
  }

  animText(): string {
    const a: AnimState | undefined = this.state.anim;
    if (a === undefined || a.samples === 0) {
      return 'no animation in this scene';
    }
    return 'Time ' + a.time.toFixed(1) + ' / ' + a.length.toFixed(0) + ' s';
  }

  infoText(line: number): string {
    const i: InfoState | undefined = this.state.info;
    if (i === undefined) {
      return '';
    }
    if (line === 0) {
      return 'Train batchsize: ' + i.batchSize.toFixed(0) + ', RHS samples: ' + i.numRHS.toFixed(0);
    }
    if (line === 1) {
      return 'Num spec rays: ' + i.specRays.toFixed(0) + ', ' + i.width.toString() + 'x' + i.height.toString() + ', MLPs on the ' + i.networks;
    }
    return 'Model: ' + i.model + ', train steps: ' + (this.state.trainSteps ?? 0).toString();
  }

  // ---------------------------------------------------------------------------------------------
  // The panel's building blocks.
  @Builder
  Header(args: HeaderArgs) {
    Row() {
      Text((this.isOpen(args.section) ? '▼ ' : '▶ ') + args.title)
        .fontSize(14)
        .fontWeight(FontWeight.Bold)
        .fontColor(TEXT)
    }
    .width('100%')
    .padding({ top: 8, bottom: 6 })
    .onClick(() => {
      this.toggleSection(args.section);
    })
  }

  @Builder
  Line(args: LineArgs) {
    Text(args.text)
      .fontSize(12)
      .fontColor(DIM)
      .width('100%')
      .padding({ top: 2, bottom: 2 })
  }

  @Builder
  Check(label: string, name: string, capability: string) {
    Row() {
      Text(label)
        .fontSize(13)
        .fontColor(this.cap(capability) ? TEXT : DIM)
        .layoutWeight(1)
      Toggle({ type: ToggleType.Switch, isOn: this.value(name) !== 0 })
        .selectedColor(ACCENT)
        .enabled(this.cap(capability))
        .onChange((on: boolean) => {
          if ((this.value(name) !== 0) !== on) {
            this.setValue(name, on ? 1 : 0);
          }
        })
    }
    .width('100%')
    .height(34)
  }

  @Builder
  Stepper(label: string, name: string, step: number, low: number, high: number, digits: number) {
    Row() {
      Text(label)
        .fontSize(13)
        .fontColor(TEXT)
        .layoutWeight(1)
      Button('-')
        .fontSize(16)
        .width(36)
        .height(28)
        .padding(0)
        .backgroundColor(IDLE)
        .onClick(() => {
          this.setValue(name, Math.max(low, this.value(name) - step));
        })
      Text(this.value(name).toFixed(digits))
        .fontSize(13)
        .fontColor(TEXT)
        .width(64)
        .textAlign(TextAlign.Center)
      Button('+')
        .fontSize(16)
        .width(36)
        .height(28)
        .padding(0)
        .backgroundColor(IDLE)
        .onClick(() => {
          this.setValue(name, Math.min(high, this.value(name) + step));
        })
    }
    .width('100%')
    .height(34)
  }

  @Builder
  Choice(label: string, name: string, options: string[]) {
    Column() {
      Text(label)
        .fontSize(13)
        .fontColor(TEXT)
        .width('100%')
      Flex({ wrap: FlexWrap.Wrap }) {
        ForEach(options, (option: string, index: number) => {
          Button(option)
            .fontSize(12)
            .height(28)
            .margin({ right: 4, top: 4 })
            .padding({ left: 8, right: 8 })
            .backgroundColor(this.value(name) === index ? ACCENT : IDLE)
            .onClick(() => {
              this.setValue(name, index);
            })
        }, (option: string) => option)
      }
      .width('100%')
    }
    .width('100%')
    .padding({ top: 4, bottom: 4 })
  }

  @Builder
  Action(label: string, command: string, capability: string) {
    Button(label)
      .fontSize(12)
      .height(28)
      .margin({ right: 4, top: 4 })
      .padding({ left: 8, right: 8 })
      .backgroundColor(IDLE)
      .enabled(this.cap(capability))
      .onClick(() => {
        entry.command(command, '');
      })
  }

  @Builder
  ScenePart() {
    Column() {
      Flex({ wrap: FlexWrap.Wrap }) {
        ForEach(this.state.scenes ?? [], (scene: SceneInfo) => {
          Button(scene.name)
            .fontSize(12)
            .height(28)
            .margin({ right: 4, top: 4 })
            .padding({ left: 8, right: 8 })
            .backgroundColor(scene.name === (this.state.scene ?? '') ? ACCENT : IDLE)
            .onClick(() => {
              this.pickScene(scene.name, this.state.variant ?? 'static');
            })
        }, (scene: SceneInfo) => scene.name)
      }
      .width('100%')

      Row() {
        Text('Variant')
          .fontSize(13)
          .fontColor(TEXT)
          .layoutWeight(1)
        ForEach(['static', 'anim'], (variant: string) => {
          Button(variant)
            .fontSize(12)
            .height(28)
            .margin({ left: 4 })
            .padding({ left: 10, right: 10 })
            .backgroundColor(variant === (this.state.variant ?? '') ? ACCENT : IDLE)
            .enabled(this.variantsOf(this.state.scene ?? '').indexOf(variant) >= 0)
            .onClick(() => {
              this.pickScene(this.state.scene ?? '', variant);
            })
        }, (variant: string) => variant)
      }
      .width('100%')
      .height(38)

      this.Check('Camera Animated', 'cameraAnimated', 'animated')
      this.Check('Play Animation', 'animate', 'animated')
      this.Stepper('Time Scale', 'timeScale', 0.25, 0, 4, 2)
      this.Line({ text: this.animText() })
      Flex({ wrap: FlexWrap.Wrap }) {
        this.Action('Reset Camera', 'resetCamera', '')
        this.Action('Rescan Scenes', 'rescan', '')
      }
      .width('100%')
      this.Line({ text: 'Drag: look around. Two fingers: move. Pinch: forward / back (camera not animated).' })
    }
    .width('100%')
  }

  @Builder
  PassPart() {
    Column() {
      this.Line({ text: 'Frame count: ' + (this.state.frame ?? 0).toString() })
      this.Choice('Render Mode', 'mode', MODES)
      Flex({ wrap: FlexWrap.Wrap }) {
        this.Action('Save model state', 'saveModel', 'train')
        this.Action('Load model state', 'loadModel', 'savedModel')
        this.Action('Reset model', 'resetModel', '')
      }
      .width('100%')
      this.Line({ text: this.infoText(0) })
      this.Line({ text: this.infoText(1) })
      this.Line({ text: this.infoText(2) })
      this.Check('Use NEE', 'useNEE', 'lights')
      this.Stepper('RHS Glossy Bounces', 'rhsGlossyBounces', 1, 0, 8, 0)
      this.Check('Diffuse Trains On Glossy', 'diffTrainOnGlossy', '')
      this.Check('Mirror Albedo', 'mirrorAlbedo', '')
      this.Choice('Env Color Encoding', 'envColorEncoding', ENV_ENCODINGS)
      this.Check('Post Filter', 'postFilter', 'postFilter')
      this.Check('FXAA', 'fxaa', 'fxaa')
    }
    .width('100%')
  }

  @Builder
  SchedulePart() {
    Column() {
      this.Check('Enable', 'trainSchedule', '')
      this.Line({ text: this.scheduleText() })
      this.Line({ text: this.rateText() })
      this.Stepper('LR hot', 'trainLrHot', 0.0005, 0.00001, 0.1, 5)
      this.Stepper('LR cold', 'trainLrCold', 0.0005, 0.00001, 0.1, 5)
      this.Stepper('EMA hot', 'trainEmaHot', 0.005, 0, 0.9999, 4)
      this.Stepper('EMA cold', 'trainEmaCold', 0.005, 0, 0.9999, 4)
      this.Stepper('Warmup frames', 'trainWarmupFrames', 50, 0, 10000, 0)
      this.Stepper('Hold frames', 'trainHoldFrames', 5, 0, 10000, 0)
      this.Stepper('Cool frames', 'trainCoolFrames', 25, 1, 10000, 0)
      this.Stepper('Camera heat', 'trainCameraHeat', 0.05, 0, 1, 2)
      this.Stepper('Scene heat', 'trainSceneHeat', 0.05, 0, 1, 2)
      this.Stepper('Light motion heat', 'trainLightMotionHeat', 0.05, 0, 1, 2)
      this.Stepper('Object motion heat', 'trainObjectMotionHeat', 0.05, 0, 1, 2)
      Flex({ wrap: FlexWrap.Wrap }) {
        this.Action('Reset schedule', 'resetSchedule', '')
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  StatsPart() {
    Column() {
      this.Line({ text: 'Frame ' + (this.state.frameMs ?? 0).toFixed(1) + ' ms + present ' + (this.state.presentMs ?? 0).toFixed(1) + ' ms' })
      ForEach(this.state.timings ?? [], (timing: Timing, index: number) => {
        Row() {
          Text(timing.ms.toFixed(1))
            .fontSize(11)
            .fontColor(TEXT)
            .width(48)
            .textAlign(TextAlign.End)
          Text(timing.name)
            .fontSize(11)
            .fontColor(DIM)
            .layoutWeight(1)
            .margin({ left: 6 })
        }
        .width('100%')
        .alignItems(VerticalAlign.Top)
        .padding({ top: 1, bottom: 1 })
      }, (timing: Timing, index: number) => index.toString() + timing.name)
    }
    .width('100%')
  }

  @Builder
  Panel() {
    Scroll() {
      Column() {
        this.Header({ title: 'Scene: ' + this.sceneText(), section: 'scene' })
        if (this.showScene) {
          this.ScenePart()
        }
        if (this.loaded()) {
          this.Header({ title: 'NeuralRadiosity', section: 'pass' })
          if (this.showPass) {
            this.PassPart()
          }
          this.Header({ title: 'Training Schedule', section: 'schedule' })
          if (this.showSchedule) {
            this.SchedulePart()
          }
          this.Header({ title: 'Frame Times', section: 'stats' })
          if (this.showStats) {
            this.StatsPart()
          }
        }
        if ((this.state.message ?? '').length > 0) {
          this.Line({ text: this.state.message ?? '' })
        }
      }
      .width('100%')
      // At least as tall as the panel: a shorter column would be centred, and the Scroll would keep
      // that offset once the column grows (the panel then opened scrolled half way down).
      .constraintSize({ minHeight: '100%' })
      .justifyContent(FlexAlign.Start)
      .padding({ left: 12, right: 12, bottom: 16 })
    }
    .width(330)
    .height('100%')
    .scrollBar(BarState.Auto)
    .backgroundColor('#C0181818')
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

      // Over the picture: the panel on the right, the status line and the panel's switch on the left.
      Row() {
        Column() {
          Row() {
            Button(this.showPanel ? 'Hide UI' : 'Show UI')
              .fontSize(12)
              .height(28)
              .padding({ left: 10, right: 10 })
              .backgroundColor('#C0303030')
              .onClick(() => {
                this.showPanel = !this.showPanel;
              })
            Text(this.frameText())
              .fontSize(12)
              .fontColor(Color.White)
              .backgroundColor('#99000000')
              .padding(4)
              .margin({ left: 8 })
          }
          Text(this.state.status)
            .fontSize(11)
            .fontColor(Color.White)
            .backgroundColor('#99000000')
            .padding(4)
            .margin({ top: 6 })
            .constraintSize({ maxWidth: 420 })
        }
        .alignItems(HorizontalAlign.Start)
        .margin(8)
        .hitTestBehavior(HitTestMode.Transparent)

        Blank()
        if (this.showPanel) {
          this.Panel()
        }
      }
      .width('100%')
      .height('100%')
      .alignItems(VerticalAlign.Top)
      .hitTestBehavior(HitTestMode.Transparent)
    }
    .width('100%')
    .height('100%')
    .backgroundColor(Color.Black)
  }
}
