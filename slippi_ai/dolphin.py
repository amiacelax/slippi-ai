import abc
import atexit
import dataclasses
import logging
import os
import subprocess
from typing import Dict, Mapping, Optional, Iterator

import fancyflags as ff
import portpicker

import melee
from melee.console import (
   DumpConfig,
   DolphinBuild,
   DolphinVersion,
   default_dolphin_install_path,
   get_exe_path,
)
import melee.console as melee_console


def _get_dolphin_version(path: str, timeout: float = 10.0) -> DolphinVersion:
  """Like melee.get_dolphin_version, but never hangs forever.

  RunPod / Docker often make `dolphin --version` hang or blow up on env size.
  ExiAI Ishiiruka returns exit code 1 for --version.
  """
  exe_path = get_exe_path(path)
  try:
    result = subprocess.run(
        [exe_path, '--version'],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
  except subprocess.TimeoutExpired:
    mainline = 'mainline' in exe_path.lower()
    logging.warning(
        'dolphin --version timed out after %ss; assuming ExiAI (%s)',
        timeout,
        'mainline' if mainline else 'Ishiiruka',
    )
    return DolphinVersion(
        mainline=mainline, version='unknown', build=DolphinBuild.EXI_AI)

  # Linux ExiAI Ishiiruka
  if result.returncode == 1:
    output = (result.stderr or result.stdout or '').strip()
    contents = output.split(' - ')
    if len(contents) >= 3 and contents[0] == 'Faster Melee' and contents[2] == 'ExiAI':
      begin = contents[1].find('(') + 1
      end = contents[1].find(')')
      version = contents[1][begin:end] if begin > 0 and end > begin else 'unknown'
      return DolphinVersion(
          mainline=False, version=version, build=DolphinBuild.EXI_AI)
    # Still treat exit 1 as ExiAI if stderr looks empty/odd on this build.
    if 'ExiAI' in output or not output:
      logging.warning(
          'dolphin --version exit 1 (%r); assuming ExiAI', output[:200])
      return DolphinVersion(
          mainline=False, version='unknown', build=DolphinBuild.EXI_AI)

  # Linux mainline
  if result.returncode == 0:
    output = (result.stdout or '').strip()
    build = DolphinBuild.EXI_AI if 'ExiAI' in output else DolphinBuild.NETPLAY
    return DolphinVersion(mainline=True, version=output or 'unknown', build=build)

  # Linux Ishiiruka netplay
  if result.returncode == 255:
    return DolphinVersion(
        mainline=False,
        version=(result.stdout or '').strip() or 'unknown',
        build=DolphinBuild.NETPLAY,
    )

  # Last resort: strings scan (fast, no dolphin process).
  try:
    strings = subprocess.run(
        ['strings', exe_path],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if strings.returncode == 0 and 'ExiAI' in strings.stdout:
      mainline = 'mainline' in exe_path.lower() or 'mainline' in strings.stdout.lower()
      logging.warning(
          'dolphin --version failed (rc=%s); strings found ExiAI (%s)',
          result.returncode,
          'mainline' if mainline else 'Ishiiruka',
      )
      return DolphinVersion(
          mainline=mainline, version='unknown', build=DolphinBuild.EXI_AI)
  except (FileNotFoundError, subprocess.TimeoutExpired):
    pass

  raise RuntimeError(
      f'Unexpected return code {result.returncode} from dolphin: '
      f'stdout={result.stdout!r} stderr={result.stderr!r}'
  )


# Console() also calls get_dolphin_version; keep both paths non-blocking.
melee_console.get_dolphin_version = _get_dolphin_version

class Player(abc.ABC):

  @abc.abstractmethod
  def controller_type(self) -> melee.ControllerType:
    pass

class Human(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.GCN_ADAPTER

@dataclasses.dataclass
class MenuingPlayer(Player):
  """Base class for CPU and AI players, which need to menu in."""

  character: melee.Character = melee.Character.FOX
  costume: Optional[int] = None

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self, index: int) -> Dict:
    return dict(
        character_selected=self.character,
        costume=index if self.costume is None else self.costume,
    )

@dataclasses.dataclass
class CPU(MenuingPlayer):
  level: int = 9

  def menuing_kwargs(self, index: int) -> Dict:
    kwargs = super().menuing_kwargs(index)
    kwargs['cpu_level'] = self.level
    return kwargs

@dataclasses.dataclass
class AI(MenuingPlayer):
  pass

def is_menu_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]

def is_game_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)

INITIAL_FRAME = -123

class ConnectFailed(Exception):
  """Raised when we fail to connect to the console."""

class WrongCharacterSelected(Exception):
  """Raised on the initial frame if the wrong character is selected."""

class Dolphin:

  def __init__(
      self,
      path: Optional[str],
      iso: Optional[str],
      players: Mapping[int, Player],
      stage: melee.Stage = melee.Stage.FINAL_DESTINATION,
      online_delay: int = 0,  # overrides Console's default of 2
      blocking_input: bool = True,
      console_timeout: Optional[float] = None,
      slippi_port: Optional[int] = None,  # Picked automatically if None
      save_replays=False,  # Override default in Console
      env_vars: Optional[dict] = None,
      headless: bool = False,
      render: Optional[bool] = None,  # Render even when running headless.
      connect_code: Optional[str] = None,
      copy_home_directory: bool = False,
      min_slp_version: Optional[tuple[int, int, int]] = (3, 18, 0),
      enable_teams: bool = False,
      **console_kwargs,
  ) -> None:
    self._players = players
    self.stage = stage
    self.min_slp_version = min_slp_version
    self._enable_teams = enable_teams
    self._teams_toggle_frames = 0

    platform = None

    # TODO: some of this logic should be moved to Console
    # Note: leave path as None to tell libmelee to look for the iso/user.json
    version = _get_dolphin_version(path or default_dolphin_install_path()[0])

    if render is None:
      render = not headless

    if not render:
      console_kwargs.update(gfx_backend='Null')

    if headless:
      console_kwargs.update(
          disable_audio=True,
      )
      if version.mainline:
        platform = 'headless'
        # console_kwargs.update(emulation_speed=0)

      if version.build is DolphinBuild.EXI_AI:
        console_kwargs.setdefault('use_exi_inputs', True)
        console_kwargs.setdefault('enable_ffw', True)
      elif not version.mainline:
        raise ValueError(
            'Headless requires mainline dolphin or a custom dolphin build. '
            'See https://github.com/vladfi1/libmelee?tab=readme-ov-file#setup-instructions')

    slippi_port = slippi_port or portpicker.pick_unused_port()

    self.menu_helper = melee.MenuHelper()

    console = melee.Console(
        path=path,
        online_delay=online_delay,
        blocking_input=blocking_input,
        polling_mode=console_timeout is not None,
        polling_timeout=console_timeout,
        slippi_port=slippi_port,
        copy_home_directory=copy_home_directory,
        setup_gecko_codes=True,
        save_replays=save_replays,
        **console_kwargs,
    )
    atexit.register(console.stop)
    self.console = console

    self.controllers: Mapping[int, melee.Controller] = {}
    self._menuing_controllers: list[tuple[melee.Controller, CPU | AI]] = []
    self._autostart = True
    self._connect_code = connect_code

    for port, player in players.items():
      skip_controller = False

      if isinstance(player, Human):
        self._autostart = False
        # Don't overwrite user's controller config
        if copy_home_directory:
          skip_controller = True

      if skip_controller:
        continue

      controller = melee.Controller(
          console, port, player.controller_type())
      self.controllers[port] = controller

      if isinstance(player, (CPU, AI)):
        self._menuing_controllers.append((controller, player))

    console.run(
        iso_path=iso,
        environment_vars=env_vars,
        platform=platform,
    )

    logging.info('Connecting to console...')
    if not console.connect():
      logging.error(
          f"PID {os.getpid()}: failed to connect to the console"
          f" {console.temp_dir} on port {slippi_port}")

      raise ConnectFailed(f"Failed to connect to the console on port {slippi_port}.")
    logging.info('Connected to console')

    for controller in self.controllers.values():
      if not controller.connect():
        self.stop()
        raise ConnectFailed("Failed to connect the controller.")

  def next_gamestate(self) -> melee.GameState:
    gamestate = self.console.step()
    if gamestate is None:
      raise TimeoutError('Console timed out.')

    # Perform some checks at the start of the game
    if is_game_state(gamestate) and gamestate.frame == INITIAL_FRAME:
      assert self.console.slp_version_tuple is not None

      if (
        self.min_slp_version is not None
        and self.console.slp_version_tuple < self.min_slp_version
      ):
        raise RuntimeError(
          f'Slippi version {self.console.slp_version_tuple} is too old. '
          f'Minimum required is {self.min_slp_version}.')

      # Phillip doesn't work well on unfrozen stadium
      if (
        self.console.slp_version_tuple >= (3, 19, 0)
        and gamestate.stage is melee.Stage.POKEMON_STADIUM
        and not self.console.is_frozen_ps
      ):
        logging.warning('Playing on unfrozen stadium')

      # Check that we picked the desired characters
      for controller, player in self._menuing_controllers:
        gs_player = gamestate.players[controller.port]
        desired_character = player.character
        actual_character = gs_player.character
        if actual_character != desired_character:
          raise WrongCharacterSelected(
            f'Port {controller.port}: expected character '
            f'{desired_character.name}, got {actual_character.name}'
          )

    return gamestate

  def step(self) -> melee.GameState:
    gamestate = self.next_gamestate()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    # if self.console.processingtime * 1000 > 12:
    #     print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

    menu_frames = 0
    while is_menu_state(gamestate):
      for i, (controller, player) in enumerate(self._menuing_controllers):

        self.menu_helper.menu_helper_simple(
            gamestate, controller,
            stage_selected=self.stage,
            connect_code=self._connect_code,
            autostart=self._autostart and i == 0 and menu_frames > 30,
            swag=False,
            **player.menuing_kwargs(i))

      # Optional: toggle Versus Teams on CSS (Y). Fail-soft if build ignores it.
      if (
          self._enable_teams
          and not getattr(gamestate, "is_teams", False)
          and gamestate.menu_state in (
              melee.Menu.CHARACTER_SELECT,
              melee.Menu.SLIPPI_ONLINE_CSS,
          )
          and self._menuing_controllers
          and self._teams_toggle_frames < 240
      ):
        leader_ctrl = self._menuing_controllers[0][0]
        if self._teams_toggle_frames % 2 == 0:
          leader_ctrl.press_button(melee.Button.BUTTON_Y)
        else:
          leader_ctrl.release_button(melee.Button.BUTTON_Y)
        self._teams_toggle_frames += 1

      gamestate = self.next_gamestate()
      menu_frames += 1

    return gamestate

  def iter_gamestates(self, skip_menu_frames: bool = True) -> Iterator[melee.GameState]:
    while True:
      gamestate = self.next_gamestate()

      menu_frames = 0
      while is_menu_state(gamestate):
        if not skip_menu_frames:
          yield gamestate

        for i, (controller, player) in enumerate(self._menuing_controllers):

          self.menu_helper.menu_helper_simple(
              gamestate, controller,
              stage_selected=self.stage,
              connect_code=self._connect_code,
              autostart=self._autostart and i == 0 and menu_frames > 180,
              swag=False,
              **player.menuing_kwargs(i))

        gamestate = self.next_gamestate()
        menu_frames += 1

      yield gamestate

  def stop(self):
    for controller in self.controllers.values():
      controller.disconnect()
    self.console.stop()

  def __del__(self):
    self.stop()

  def multi_step(self, n: int):
    for _ in range(n):
      self.step()

_field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class DolphinConfig:
  """Configure dolphin for evaluation."""
  path: Optional[str] = None  # Path to folder containing the dolphin executable
  iso: Optional[str] = None  # Path to melee 1.02 iso.
  dolphin_home_path: Optional[str] = None  # Path to dolphin home directory.
  tmp_home_directory: bool = True  # Create a temporary home directory for dolphin.
  copy_home_directory: bool = False  # Copy the dolphin home directory to a temp location.
  stage: melee.Stage = melee.Stage.RANDOM_STAGE  # Which stage to play on.
  online_delay: int = 0  # Simulate online delay.
  blocking_input: bool = True  # Have game wait for AIs to send inputs.
  console_timeout: Optional[float] = None  # Seconds to wait for console inputs before throwing an error.
  slippi_port: Optional[int] = None  # Local ip port to communicate with dolphin.
  fullscreen: bool = False # Run dolphin in full screen mode
  render: Optional[bool] = None  # Render frames. Only disable if using vladfi1\'s slippi fork.
  save_replays: bool = False  # Save slippi replays to the usual location.
  replay_dir: Optional[str] = None  # Directory to save replays to.
  gfx_backend: str = ''  # Graphics backend to use.
  disable_audio: bool = False  # Disable dolphin audio.
  audio_backend: str = ''  # Audio backend to use.
  headless: bool = True  # Headless configuration: exi + ffw, no graphics or audio.
  emulation_speed: float = 0.0  # Set to 0 for unlimited speed. Mainline only.
  infinite_time: bool = True  # Infinite time no stocks.
  instant_match_restart: bool = True  # Skip menuing between games, start new game on random stage.
  log_level: int = 3  # WARN; 0 to disable
  log_types: list[str] = dataclasses.field(default_factory=['SLIPPI'].copy)
  dump: DumpConfig = _field(DumpConfig)  # For framedumping.

  # For online play
  connect_code: Optional[str] = None
  user_json_path: Optional[str] = None

  def to_kwargs(self) -> dict:
    kwargs = dataclasses.asdict(self)
    del kwargs['dump']
    kwargs['dump_config'] = self.dump
    return kwargs

  @classmethod
  def kwargs_from_flags(cls, flags: dict) -> dict:
    kwargs = flags.copy()
    del kwargs['dump']
    kwargs['dump_config'] = DumpConfig(**flags['dump'])
    return kwargs

# TODO: replace usage with the above dataclass
DOLPHIN_FLAGS = dict(
    path=ff.String(None, 'Path to folder containing the dolphin executable.'),
    iso=ff.String(None, 'Path to melee 1.02 iso.'),
    stage=ff.EnumClass(melee.Stage.RANDOM_STAGE, melee.Stage, 'Which stage to play on.'),
    online_delay=ff.Integer(0, 'Simulate online delay.'),
    blocking_input=ff.Boolean(True, 'Have game wait for AIs to send inputs.'),
    slippi_port=ff.Integer(None, 'Local ip port to communicate with dolphin.'),
    fullscreen=ff.Boolean(False, 'Run dolphin in full screen mode.'),
    render=ff.Boolean(None, 'Render frames. Only disable if using vladfi1\'s slippi fork.'),
    save_replays=ff.Boolean(False, 'Save slippi replays to the usual location.'),
    replay_dir=ff.String(None, 'Directory to save replays to.'),
    headless=ff.Boolean(
        False, 'Headless configuration: exi + ffw, no graphics or audio.'),
    emulation_speed=ff.Float(1.0),
    infinite_time=ff.Boolean(False, 'Infinite time no stocks.'),
    log_level=ff.Integer(3, 'Dolphin log level, defaults to WARN.'),
    log_types=ff.StringList(['SLIPPI'], 'Enabled logging categories.'),
    disable_audio=ff.Boolean(False, 'Disable dolphin audio.'),
)
