#!/usr/bin/env python3
"""
Convert a small, practical subset of PBRT-v4 scenes into Falcor .pyscene files.

This script is intentionally scoped to the subset used by
`media/scenes/veach-ajar/scene-v4.pbrt` and similar scenes:

- Transform / WorldBegin / AttributeBegin / AttributeEnd
- Camera "perspective"
- Film / Sampler / PixelFilter / Integrator (parsed, mostly emitted as comments)
- Texture "...\" \"float|spectrum\" \"imagemap\""
- MakeNamedMaterial / NamedMaterial / Material
- AreaLightSource "diffuse"
- Shape "plymesh"
- Shape "trianglemesh"

The matrix handling mirrors Falcor's PBRT importer:
- PBRT files store row-vector transforms, so each 4x4 transform is transposed.
- Camera transforms are converted using inverse(cameraFromWorld) * kInvertZ.

References:
- PBRT v4 file format: https://www.pbrt.org/fileformat-v4
- Falcor PBRT importer:
  Source/plugins/importers/PBRTImporter/Builder.cpp
  Source/plugins/importers/PBRTImporter/PBRTImporter.cpp
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, NoReturn


TOKEN_RE = re.compile(r'"[^"]*"|\[|\]|[^\s\[\]"]+')
K_INVERT_Z = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
NAMED_RGB_SPECTRA = {
    "metal-Ag-eta": [1.65746, 0.880369, 0.521229],
    "metal-Ag-k": [9.223869, 6.269523, 4.837001],
}


def fail(message: str, source_path: Optional[Path] = None, line: Optional[int] = None) -> NoReturn:
    location = ""
    if source_path is not None:
        location = source_path.as_posix()
        if line is not None:
            location += f":{line}"
        location += ": "
    raise SystemExit(f"error: {location}{message}")


@dataclass
class Token:
    text: str
    line: int


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        hash_index = line.find("#")
        if hash_index != -1:
            line = line[:hash_index]
        lines.append(line)
    return "\n".join(lines)


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        uncommented = line.split("#", 1)[0]
        for match in TOKEN_RE.finditer(uncommented):
            tokens.append(Token(text=match.group(0), line=line_number))
    return tokens


def is_quoted(token: str) -> bool:
    return len(token) >= 2 and token[0] == '"' and token[-1] == '"'


def unquote(token: str) -> str:
    return token[1:-1]


def parse_scalar(token: str):
    if is_quoted(token):
        return unquote(token)
    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(ch in token for ch in ".eE"):
            return float(token)
        return int(token)
    except ValueError:
        return token


def identity_matrix() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat_mul(a: Iterable[Iterable[float]], b: Iterable[Iterable[float]]) -> List[List[float]]:
    a_rows = [list(row) for row in a]
    b_rows = [list(row) for row in b]
    out = [[0.0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            out[r][c] = sum(a_rows[r][k] * b_rows[k][c] for k in range(4))
    return out


def transpose_pbrt_matrix(values: List[float]) -> List[List[float]]:
    if len(values) != 16:
        fail(f"expected 16 matrix values, got {len(values)}")
    return [
        [values[0], values[4], values[8], values[12]],
        [values[1], values[5], values[9], values[13]],
        [values[2], values[6], values[10], values[14]],
        [values[3], values[7], values[11], values[15]],
    ]


def affine_inverse(m: List[List[float]]) -> List[List[float]]:
    r = [[m[row][col] for col in range(3)] for row in range(3)]
    t = [m[row][3] for row in range(3)]
    rt = [[r[col][row] for col in range(3)] for row in range(3)]
    inv_t = [-(rt[row][0] * t[0] + rt[row][1] * t[1] + rt[row][2] * t[2]) for row in range(3)]
    return [
        [rt[0][0], rt[0][1], rt[0][2], inv_t[0]],
        [rt[1][0], rt[1][1], rt[1][2], inv_t[1]],
        [rt[2][0], rt[2][1], rt[2][2], inv_t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def determinant3(m: List[List[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def quat_from_matrix(m: List[List[float]]) -> List[float]:
    four_x_squared_minus_1 = m[0][0] - m[1][1] - m[2][2]
    four_y_squared_minus_1 = m[1][1] - m[0][0] - m[2][2]
    four_z_squared_minus_1 = m[2][2] - m[0][0] - m[1][1]
    four_w_squared_minus_1 = m[0][0] + m[1][1] + m[2][2]

    biggest_index = 0
    four_biggest_squared_minus_1 = four_w_squared_minus_1
    if four_x_squared_minus_1 > four_biggest_squared_minus_1:
        four_biggest_squared_minus_1 = four_x_squared_minus_1
        biggest_index = 1
    if four_y_squared_minus_1 > four_biggest_squared_minus_1:
        four_biggest_squared_minus_1 = four_y_squared_minus_1
        biggest_index = 2
    if four_z_squared_minus_1 > four_biggest_squared_minus_1:
        four_biggest_squared_minus_1 = four_z_squared_minus_1
        biggest_index = 3

    biggest_val = math.sqrt(four_biggest_squared_minus_1 + 1.0) * 0.5
    mult = 0.25 / biggest_val

    if biggest_index == 0:
        w = biggest_val
        x = (m[1][2] - m[2][1]) * mult
        y = (m[2][0] - m[0][2]) * mult
        z = (m[0][1] - m[1][0]) * mult
    elif biggest_index == 1:
        x = biggest_val
        w = (m[1][2] - m[2][1]) * mult
        y = (m[0][1] + m[1][0]) * mult
        z = (m[2][0] + m[0][2]) * mult
    elif biggest_index == 2:
        y = biggest_val
        w = (m[2][0] - m[0][2]) * mult
        x = (m[0][1] + m[1][0]) * mult
        z = (m[1][2] + m[2][1]) * mult
    else:
        z = biggest_val
        w = (m[0][1] - m[1][0]) * mult
        x = (m[2][0] + m[0][2]) * mult
        y = (m[1][2] + m[2][1]) * mult

    return [x, y, z, w]


def euler_from_quat(q: List[float]) -> List[float]:
    x, y, z, w = q

    pitch_y = 2.0 * (y * z + w * x)
    pitch_x = w * w - x * x - y * y + z * z
    if abs(pitch_x) < 1e-12 and abs(pitch_y) < 1e-12:
        pitch = 2.0 * math.atan2(x, w)
    else:
        pitch = math.atan2(pitch_y, pitch_x)

    yaw = math.asin(max(-1.0, min(1.0, -2.0 * (x * z - w * y))))
    roll = math.atan2(2.0 * (x * y + w * z), w * w + x * x - y * y - z * z)
    return [math.degrees(pitch), math.degrees(yaw), math.degrees(roll)]


def decompose_transform(m: List[List[float]]) -> tuple[List[float], List[float], List[float]]:
    translation = [m[0][3], m[1][3], m[2][3]]
    cols = [[m[0][i], m[1][i], m[2][i]] for i in range(3)]
    scales = [math.sqrt(sum(v * v for v in col)) for col in cols]

    if any(scale < 1e-8 for scale in scales):
        fail("encountered a singular transform that cannot be converted to Falcor Transform()")

    rot_cols = [[col[i] / scales[idx] for i in range(3)] for idx, col in enumerate(cols)]
    rot = [[rot_cols[col][row] for col in range(3)] for row in range(3)]
    if determinant3(rot) < 0.0:
        axis = max(range(3), key=lambda i: abs(scales[i]))
        scales[axis] *= -1.0
        rot_cols[axis] = [-v for v in rot_cols[axis]]
        rot = [[rot_cols[col][row] for col in range(3)] for row in range(3)]

    # Falcor's Transform.rotationEulerDeg uses the inverse sign convention
    # relative to the Euler angles extracted here from the rotation matrix.
    # Negating the angles makes the emitted TRS reconstruct the original matrix.
    euler_deg = [-value for value in euler_from_quat(quat_from_matrix(rot))]
    return translation, scales, euler_deg


def convert_camera_transform(camera_from_world: List[List[float]]) -> tuple[List[float], List[float], List[float]]:
    world_from_camera = affine_inverse(camera_from_world)
    final = mat_mul(world_from_camera, K_INVERT_Z)
    position = [final[0][3], final[1][3], final[2][3]]
    up = [final[0][1], final[1][1], final[2][1]]
    forward = [-final[0][2], -final[1][2], -final[2][2]]
    target = [position[i] + forward[i] for i in range(3)]
    return position, target, up


def fov_y_to_focal_length_mm(fov_degrees: float, frame_height_mm: float = 24.0) -> float:
    fov_radians = math.radians(fov_degrees)
    if abs(fov_radians) < 1e-12:
        return 0.0
    return frame_height_mm / (2.0 * math.tan(fov_radians * 0.5))


def sanitize_identifier(name: str, prefix: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = prefix
    if value[0].isdigit():
        value = f"{prefix}_{value}"
    return value


def format_number(value: float) -> str:
    if abs(value) < 1e-9:
        value = 0.0
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):.1f}"
    return f"{value:.9g}"


def format_float2(values: List[float]) -> str:
    return f"float2({format_number(values[0])}, {format_number(values[1])})"


def format_float3(values: List[float]) -> str:
    return f"float3({format_number(values[0])}, {format_number(values[1])}, {format_number(values[2])})"


def format_float4(values: List[float]) -> str:
    return f"float4({format_number(values[0])}, {format_number(values[1])}, {format_number(values[2])}, {format_number(values[3])})"


def is_close_vec(values: List[float], expected: List[float], eps: float = 1e-5) -> bool:
    return all(abs(a - b) <= eps for a, b in zip(values, expected))


def format_transform_expr(matrix: List[List[float]]) -> str:
    translation, scaling, rotation_euler_deg = decompose_transform(matrix)
    args = []
    if not is_close_vec(scaling, [1.0, 1.0, 1.0]):
        args.append(f"scaling={format_float3(scaling)}")
    if not is_close_vec(translation, [0.0, 0.0, 0.0]):
        args.append(f"translation={format_float3(translation)}")
    if not is_close_vec(rotation_euler_deg, [0.0, 0.0, 0.0], eps=1e-4):
        args.append(f"rotationEulerDeg={format_float3(rotation_euler_deg)}")
    if not args:
        return "Transform()"
    return f"Transform({', '.join(args)})"


def envmap_rotation_from_transform(matrix: List[List[float]]) -> List[float]:
    _, _, rotation_euler_deg = decompose_transform(matrix)
    return [-value for value in rotation_euler_deg]


@dataclass
class Param:
    kind: str
    value: object
    line: int


@dataclass
class TextureDef:
    name: str
    value_type: str
    texture_type: str
    params: Dict[str, Param]
    line: int


@dataclass
class MaterialDef:
    name: str
    material_type: str
    params: Dict[str, Param]
    line: int


@dataclass
class AreaLightDef:
    light_type: str
    params: Dict[str, Param]
    line: int


@dataclass
class InfiniteLightDef:
    params: Dict[str, Param]
    transform: List[List[float]]
    line: int


@dataclass
class ShapeDef:
    shape_type: str
    params: Dict[str, Param]
    transform: List[List[float]]
    material_name: Optional[str]
    area_light: Optional[AreaLightDef]
    index: int
    line: int


@dataclass
class SceneData:
    source_path: Optional[Path] = None
    camera_transform: List[List[float]] = field(default_factory=identity_matrix)
    camera_params: Dict[str, Param] = field(default_factory=dict)
    film_params: Dict[str, Param] = field(default_factory=dict)
    integrator_name: Optional[str] = None
    sampler_name: Optional[str] = None
    pixel_filter_name: Optional[str] = None
    textures: Dict[str, TextureDef] = field(default_factory=dict)
    materials: Dict[str, MaterialDef] = field(default_factory=dict)
    infinite_lights: List[InfiniteLightDef] = field(default_factory=list)
    shapes: List[ShapeDef] = field(default_factory=list)


@dataclass
class GraphicsState:
    transform: List[List[float]] = field(default_factory=identity_matrix)
    current_material: Optional[str] = None
    area_light: Optional[AreaLightDef] = None


class TokenStream:
    def __init__(self, tokens: List[Token], source_path: Path):
        self.tokens = tokens
        self.index = 0
        self.source_path = source_path

    def empty(self) -> bool:
        return self.index >= len(self.tokens)

    def peek_token(self) -> Optional[Token]:
        if self.empty():
            return None
        return self.tokens[self.index]

    def peek(self) -> Optional[str]:
        token = self.peek_token()
        return token.text if token is not None else None

    def pop_token(self) -> Token:
        if self.empty():
            fail("unexpected end of file", self.source_path)
        token = self.tokens[self.index]
        self.index += 1
        return token

    def pop(self) -> str:
        return self.pop_token().text

    def expect(self, expected: str) -> None:
        token = self.pop_token()
        if token.text != expected:
            fail(f"expected '{expected}', got '{token.text}'", self.source_path, token.line)

    def parse_param_list(self) -> Dict[str, Param]:
        params: Dict[str, Param] = {}
        while True:
            token = self.peek_token()
            if token is None or not is_quoted(token.text):
                break

            typed_name_token = self.pop_token()
            typed_name = unquote(typed_name_token.text)
            if " " not in typed_name:
                fail(f"invalid parameter declaration '{typed_name}'", self.source_path, typed_name_token.line)

            kind, name = typed_name.split(" ", 1)
            if self.peek() == "[":
                self.pop()
                values = []
                while self.peek() != "]":
                    if self.peek() is None:
                        fail(f"unterminated array parameter '{name}'", self.source_path, typed_name_token.line)
                    values.append(parse_scalar(self.pop()))
                self.expect("]")
                value = values
            else:
                value = parse_scalar(self.pop())

            params[name] = Param(kind=kind, value=value, line=typed_name_token.line)
        return params


def require_param(params: Dict[str, Param], name: str, source_path: Optional[Path] = None, line: Optional[int] = None) -> Param:
    if name not in params:
        fail(f"missing required parameter '{name}'", source_path, line)
    return params[name]


def maybe_list(param: Optional[Param]) -> List[object]:
    if param is None:
        return []
    if isinstance(param.value, list):
        return param.value
    return [param.value]


def get_float(params: Dict[str, Param], name: str, default: float) -> float:
    param = params.get(name)
    if param is None:
        return default
    value = param.value
    if isinstance(value, list):
        return float(value[0])
    return float(value)


def get_bool(params: Dict[str, Param], name: str, default: bool) -> bool:
    param = params.get(name)
    if param is None:
        return default
    value = param.value
    if isinstance(value, list):
        value = value[0]
    return bool(value)


def get_string(params: Dict[str, Param], name: str, default: str = "") -> str:
    param = params.get(name)
    if param is None:
        return default
    value = param.value
    if isinstance(value, list):
        value = value[0]
    return str(value)


def get_rgb(params: Dict[str, Param], name: str, default: List[float], source_path: Optional[Path] = None, line: Optional[int] = None) -> List[float]:
    param = params.get(name)
    if param is None:
        return list(default)
    values = maybe_list(param)
    if len(values) == 1:
        value = values[0]
        if isinstance(value, str):
            rgb = NAMED_RGB_SPECTRA.get(value)
            if rgb is None:
                fail(f"unsupported named spectrum '{value}' for parameter '{name}'", source_path, param.line if param else line)
            return list(rgb)
        if isinstance(value, (int, float)):
            scalar = float(value)
            return [scalar, scalar, scalar]
    if len(values) != 3:
        fail(f"parameter '{name}' must have 3 components", source_path, param.line if param else line)
    return [float(v) for v in values]


def get_roughness_pair(params: Dict[str, Param]) -> List[float]:
    uroughness = params.get("uroughness")
    vroughness = params.get("vroughness")
    roughness = params.get("roughness")

    if uroughness is None and roughness is not None:
        uroughness_value = float(maybe_list(roughness)[0])
    else:
        uroughness_value = float(maybe_list(uroughness)[0]) if uroughness is not None else 0.0

    if vroughness is None and roughness is not None:
        vroughness_value = float(maybe_list(roughness)[0])
    else:
        vroughness_value = float(maybe_list(vroughness)[0]) if vroughness is not None else 0.0

    if get_bool(params, "remaproughness", True):
        uroughness_value = math.sqrt(uroughness_value)
        vroughness_value = math.sqrt(vroughness_value)

    return [uroughness_value, vroughness_value]


def reflectance_to_eta_k(reflectance: List[float]) -> tuple[List[float], List[float]]:
    r = [max(0.0, min(0.9999, c)) for c in reflectance]
    eta = [1.0, 1.0, 1.0]
    k = [2.0 * math.sqrt(c) / math.sqrt(1.0 - c) for c in r]
    return eta, k


def get_scalar_roughness(params: Dict[str, Param]) -> float:
    uroughness, vroughness = get_roughness_pair(params)
    return 0.5 * (uroughness + vroughness)


def get_scalar_eta(params: Dict[str, Param], default: float = 1.5) -> float:
    param = params.get("eta")
    if param is None:
        return default
    values = maybe_list(param)
    if len(values) == 1:
        value = values[0]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            rgb = NAMED_RGB_SPECTRA.get(value)
            if rgb is not None:
                return sum(rgb) / 3.0
    rgb = get_rgb(params, "eta", [default, default, default], None, param.line)
    return sum(rgb) / 3.0


def conductor_specular_albedo(params: Dict[str, Param], source_path: Optional[Path] = None, line: Optional[int] = None) -> List[float]:
    if "reflectance" in params:
        return get_rgb(params, "reflectance", [0.5, 0.5, 0.5], source_path, line)

    eta = get_rgb(params, "eta", [1.65746, 0.880369, 0.521229], source_path, line)
    k = get_rgb(params, "k", [9.223869, 6.269523, 4.837001], source_path, line)

    reflectance = []
    for eta_i, k_i in zip(eta, k):
        numerator = (eta_i - 1.0) * (eta_i - 1.0) + k_i * k_i
        denominator = (eta_i + 1.0) * (eta_i + 1.0) + k_i * k_i
        reflectance.append(numerator / denominator)
    return reflectance


def resolve_asset_path(asset: str, source_dir: Path, output_dir: Path) -> str:
    absolute = Path(asset) if Path(asset).is_absolute() else (source_dir / asset)
    return os.path.relpath(absolute, output_dir).replace("\\", "/")


def parse_scene(path: Path) -> SceneData:
    stream = TokenStream(tokenize(path.read_text(encoding="utf-8")), path)
    scene = SceneData(source_path=path)
    state = GraphicsState()
    stack: List[GraphicsState] = []
    world_started = False
    shape_index = 0

    while not stream.empty():
        keyword_token = stream.pop_token()
        keyword = keyword_token.text

        if keyword == "Integrator":
            scene.integrator_name = unquote(stream.pop())
            stream.parse_param_list()
        elif keyword == "Sampler":
            scene.sampler_name = unquote(stream.pop())
            stream.parse_param_list()
        elif keyword == "PixelFilter":
            scene.pixel_filter_name = unquote(stream.pop())
            stream.parse_param_list()
        elif keyword == "Film":
            stream.pop()
            scene.film_params = stream.parse_param_list()
        elif keyword == "Translate":
            # PBRT: Translate x y z  ->  state.transform = T * state.transform
            # (row-vector convention; applied to current transform like Transform)
            tx = float(stream.pop())
            ty = float(stream.pop())
            tz = float(stream.pop())
            translate = [
                [1.0, 0.0, 0.0, tx],
                [0.0, 1.0, 0.0, ty],
                [0.0, 0.0, 1.0, tz],
                [0.0, 0.0, 0.0, 1.0],
            ]
            state.transform = mat_mul(translate, state.transform)
        elif keyword == "Transform":
            stream.expect("[")
            values = []
            while stream.peek() != "]":
                if stream.peek() is None:
                    fail("unterminated Transform matrix")
                values.append(float(stream.pop()))
            stream.expect("]")
            state.transform = transpose_pbrt_matrix(values)
        elif keyword == "Camera":
            camera_type = unquote(stream.pop())
            if camera_type != "perspective":
                fail(f"unsupported camera type '{camera_type}'", path, keyword_token.line)
            scene.camera_transform = copy.deepcopy(state.transform)
            scene.camera_params = stream.parse_param_list()
        elif keyword == "WorldBegin":
            world_started = True
            state = GraphicsState()
        elif keyword == "Texture":
            if not world_started:
                fail("Texture must appear inside WorldBegin", path, keyword_token.line)
            texture_name = unquote(stream.pop())
            value_type = unquote(stream.pop())
            texture_type = unquote(stream.pop())
            params = stream.parse_param_list()
            scene.textures[texture_name] = TextureDef(texture_name, value_type, texture_type, params, keyword_token.line)
        elif keyword == "MakeNamedMaterial":
            if not world_started:
                fail("MakeNamedMaterial must appear inside WorldBegin", path, keyword_token.line)
            material_name = unquote(stream.pop())
            params = stream.parse_param_list()
            material_type = get_string(params, "type")
            if not material_type:
                fail(f"material '{material_name}' is missing string type", path, keyword_token.line)
            scene.materials[material_name] = MaterialDef(material_name, material_type, params, keyword_token.line)
        elif keyword == "Material":
            if not world_started:
                fail("Material must appear inside WorldBegin", path, keyword_token.line)
            material_type = unquote(stream.pop())
            params = stream.parse_param_list()
            material_name = f"__inline_material_{len(scene.materials)}"
            scene.materials[material_name] = MaterialDef(material_name, material_type, params, keyword_token.line)
            state.current_material = material_name
        elif keyword == "NamedMaterial":
            state.current_material = unquote(stream.pop())
        elif keyword == "AreaLightSource":
            light_type = unquote(stream.pop())
            params = stream.parse_param_list()
            state.area_light = AreaLightDef(light_type=light_type, params=params, line=keyword_token.line)
        elif keyword == "LightSource":
            if not world_started:
                fail("LightSource must appear inside WorldBegin", path, keyword_token.line)
            light_type = unquote(stream.pop())
            params = stream.parse_param_list()
            if light_type != "infinite":
                fail(f"unsupported light source type '{light_type}'", path, keyword_token.line)
            scene.infinite_lights.append(
                InfiniteLightDef(
                    params=params,
                    transform=copy.deepcopy(state.transform),
                    line=keyword_token.line,
                )
            )
        elif keyword == "AttributeBegin":
            stack.append(copy.deepcopy(state))
        elif keyword == "AttributeEnd":
            if not stack:
                fail("unmatched AttributeEnd", path, keyword_token.line)
            state = stack.pop()
        elif keyword == "Shape":
            shape_type = unquote(stream.pop())
            if shape_type not in {"plymesh", "trianglemesh"}:
                fail(f"unsupported shape type '{shape_type}'", path, keyword_token.line)
            shape = ShapeDef(
                shape_type=shape_type,
                params=stream.parse_param_list(),
                transform=copy.deepcopy(state.transform),
                material_name=state.current_material,
                area_light=copy.deepcopy(state.area_light),
                index=shape_index,
                line=keyword_token.line,
            )
            scene.shapes.append(shape)
            shape_index += 1
        else:
            fail(f"unsupported PBRT statement '{keyword}' in this converter", path, keyword_token.line)

    if stack:
        fail("missing AttributeEnd", path)
    return scene


class Emitter:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self._used: Dict[str, int] = {}

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def unique(self, base: str) -> str:
        base = sanitize_identifier(base, "v")
        index = self._used.get(base, 0)
        self._used[base] = index + 1
        if index == 0:
            return base
        return f"{base}_{index}"

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def emit_material(
    emitter: Emitter,
    material: MaterialDef,
    textures: Dict[str, TextureDef],
    source_dir: Path,
    output_dir: Path,
    source_path: Path,
    use_pbrt_materials: bool,
) -> str:
    material_var = emitter.unique(material.name)
    material_type = material.material_type
    params = material.params

    if material_type == "diffuse":
        if use_pbrt_materials:
            emitter.add(f"{material_var} = PBRTDiffuseMaterial({material.name!r})")
        else:
            emitter.add(f"{material_var} = StandardMaterial({material.name!r})")
            emitter.add(f"{material_var}.metallic = 0.0")
            emitter.add(f"{material_var}.roughness = 1.0")
        if "reflectance" in params and params["reflectance"].kind == "texture":
            texture_name = get_string(params, "reflectance")
            texture = textures.get(texture_name)
            if texture is None:
                fail(f"material '{material.name}' references unknown texture '{texture_name}'", source_path, params["reflectance"].line)
            if texture.texture_type != "imagemap":
                fail(f"texture '{texture_name}' uses unsupported type '{texture.texture_type}'", source_path, texture.line)
            filename = resolve_asset_path(get_string(texture.params, "filename"), source_dir, output_dir)
            emitter.add(f"{material_var}.loadTexture(MaterialTextureSlot.BaseColor, {filename!r})")
        else:
            emitter.add(
                f"{material_var}.baseColor = {format_float4(get_rgb(params, 'reflectance', [0.5, 0.5, 0.5], source_path, material.line) + [1.0])}"
            )
        emitter.add(f"{material_var}.doubleSided = True")
    elif material_type == "coateddiffuse":
        if use_pbrt_materials:
            emitter.add(f"{material_var} = PBRTCoatedDiffuseMaterial({material.name!r})")
            emitter.add(f"{material_var}.roughness = {format_float2(get_roughness_pair(params))}")
        else:
            emitter.add(f"{material_var} = StandardMaterial({material.name!r})")
            emitter.add(f"{material_var}.metallic = 0.0")
            emitter.add(f"{material_var}.roughness = {format_number(math.sqrt(get_scalar_roughness(params)))}")
        if "reflectance" in params and params["reflectance"].kind == "texture":
            texture_name = get_string(params, "reflectance")
            texture = textures.get(texture_name)
            if texture is None:
                fail(f"material '{material.name}' references unknown texture '{texture_name}'", source_path, params["reflectance"].line)
            filename = resolve_asset_path(get_string(texture.params, "filename"), source_dir, output_dir)
            emitter.add(f"{material_var}.loadTexture(MaterialTextureSlot.BaseColor, {filename!r})")
        else:
            emitter.add(
                f"{material_var}.baseColor = {format_float4(get_rgb(params, 'reflectance', [0.5, 0.5, 0.5], source_path, material.line) + [1.0])}"
            )
        emitter.add(f"{material_var}.doubleSided = True")
    elif material_type == "conductor":
        if "reflectance" in params:
            eta, k = reflectance_to_eta_k(get_rgb(params, "reflectance", [0.5, 0.5, 0.5], source_path, material.line))
        else:
            eta = get_rgb(params, "eta", [1.65746, 0.880369, 0.521229], source_path, material.line)
            k = get_rgb(params, "k", [9.223869, 6.269523, 4.837001], source_path, material.line)
        if use_pbrt_materials:
            emitter.add(f"{material_var} = PBRTConductorMaterial({material.name!r})")
            emitter.add(f"{material_var}.baseColor = {format_float4(eta + [1.0])}")
            emitter.add(f"{material_var}.transmissionColor = {format_float3(k)}")
            emitter.add(f"{material_var}.roughness = {format_float2(get_roughness_pair(params))}")
        else:
            emitter.add(f"{material_var} = StandardMaterial({material.name!r})")
            emitter.add(f"{material_var}.baseColor = {format_float4(conductor_specular_albedo(params, source_path, material.line) + [1.0])}")
            emitter.add(f"{material_var}.metallic = 1.0")
            emitter.add(f"{material_var}.roughness = {format_number(math.sqrt(get_scalar_roughness(params)))}")
        emitter.add(f"{material_var}.doubleSided = True")
    elif material_type == "dielectric":
        eta = get_scalar_eta(params, 1.5)
        if use_pbrt_materials:
            emitter.add(f"{material_var} = PBRTDielectricMaterial({material.name!r})")
            emitter.add(f"{material_var}.indexOfRefraction = {format_number(eta)}")
            emitter.add(f"{material_var}.roughness = {format_float2(get_roughness_pair(params))}")
        else:
            emitter.add(f"{material_var} = StandardMaterial({material.name!r})")
            emitter.add(f"{material_var}.metallic = 0.0")
            emitter.add(f"{material_var}.roughness = {format_number(math.sqrt(get_scalar_roughness(params)))}")
            emitter.add(f"{material_var}.indexOfRefraction = {format_number(eta)}")
            emitter.add(f"{material_var}.specularTransmission = 1.0")
    elif material_type == "diffusetransmission":
        if use_pbrt_materials:
            emitter.add(f"{material_var} = PBRTDiffuseTransmissionMaterial({material.name!r})")
        else:
            emitter.add(f"{material_var} = StandardMaterial({material.name!r})")
            emitter.add(f"{material_var}.metallic = 0.0")
            emitter.add(f"{material_var}.roughness = 1.0")
            emitter.add(f"{material_var}.diffuseTransmission = 0.5")
        if "reflectance" in params and params["reflectance"].kind == "texture":
            texture_name = get_string(params, "reflectance")
            texture = textures.get(texture_name)
            if texture is None:
                fail(f"material '{material.name}' references unknown texture '{texture_name}'", source_path, params["reflectance"].line)
            filename = resolve_asset_path(get_string(texture.params, "filename"), source_dir, output_dir)
            emitter.add(f"{material_var}.loadTexture(MaterialTextureSlot.BaseColor, {filename!r})")
        else:
            emitter.add(
                f"{material_var}.baseColor = {format_float4(get_rgb(params, 'reflectance', [0.25, 0.25, 0.25], source_path, material.line) + [1.0])}"
            )

        if "transmittance" in params and params["transmittance"].kind == "texture":
            texture_name = get_string(params, "transmittance")
            texture = textures.get(texture_name)
            if texture is None:
                fail(f"material '{material.name}' references unknown texture '{texture_name}'", source_path, params["transmittance"].line)
            filename = resolve_asset_path(get_string(texture.params, "filename"), source_dir, output_dir)
            emitter.add(f"{material_var}.loadTexture(MaterialTextureSlot.Transmission, {filename!r})")
        else:
            emitter.add(
                f"{material_var}.transmissionColor = {format_float3(get_rgb(params, 'transmittance', [0.25, 0.25, 0.25], source_path, material.line))}"
            )

        emitter.add(f"{material_var}.doubleSided = True")
    else:
        fail(f"unsupported material type '{material_type}'", source_path, material.line)

    return material_var


def emit_area_light_material(emitter: Emitter, shape: ShapeDef, source_material_name: Optional[str], source_path: Path) -> str:
    if shape.area_light is None:
        fail("internal error: missing area light", source_path, shape.line)
    if shape.area_light.light_type != "diffuse":
        fail(f"unsupported area light type '{shape.area_light.light_type}'", source_path, shape.area_light.line)

    base_name = source_material_name or f"AreaLight{shape.index}"
    material_var = emitter.unique(f"{base_name}_emissive")
    emitter.add(f"{material_var} = StandardMaterial({base_name!r})")
    emitter.add(f"{material_var}.baseColor = float4(0.0, 0.0, 0.0, 1.0)")
    emitter.add(f"{material_var}.roughness = 0.0")
    emitter.add(
        f"{material_var}.emissiveColor = {format_float3(get_rgb(shape.area_light.params, 'L', [1.0, 1.0, 1.0], source_path, shape.area_light.line))}"
    )
    emitter.add(f"{material_var}.doubleSided = True")
    return material_var


def emit_trianglemesh_inline(emitter: Emitter, shape: ShapeDef, source_path: Path) -> str:
    mesh_var = emitter.unique(f"trianglemesh_{shape.index}")
    params = shape.params
    p_values = maybe_list(require_param(params, "P", source_path, shape.line))
    index_values = maybe_list(require_param(params, "indices", source_path, shape.line))
    uv_values = maybe_list(params.get("uv"))
    n_values = maybe_list(params.get("N"))

    if len(p_values) % 3 != 0:
        fail(f"trianglemesh {shape.index} has invalid point3 P data", source_path, shape.line)
    if len(uv_values) not in {0, (len(p_values) // 3) * 2}:
        fail(f"trianglemesh {shape.index} has invalid point2 uv data", source_path, shape.line)
    if len(n_values) not in {0, len(p_values)}:
        fail(f"trianglemesh {shape.index} has invalid normal N data", source_path, shape.line)
    if len(index_values) % 3 != 0:
        fail(f"trianglemesh {shape.index} has invalid indices data", source_path, shape.line)

    positions = [[float(p_values[i]), float(p_values[i + 1]), float(p_values[i + 2])] for i in range(0, len(p_values), 3)]
    normals = (
        [[0.0, 0.0, 0.0] for _ in positions]
        if not n_values
        else [[float(n_values[i]), float(n_values[i + 1]), float(n_values[i + 2])] for i in range(0, len(n_values), 3)]
    )
    uvs = (
        [[0.0, 0.0] for _ in positions]
        if not uv_values
        else [[float(uv_values[i]), float(uv_values[i + 1])] for i in range(0, len(uv_values), 2)]
    )

    emitter.add(f"{mesh_var} = TriangleMesh()")
    for position, normal, uv in zip(positions, normals, uvs):
        emitter.add(f"{mesh_var}.addVertex({format_float3(position)}, {format_float3(normal)}, {format_float2(uv)})")
    for i in range(0, len(index_values), 3):
        emitter.add(f"{mesh_var}.addTriangle({int(index_values[i])}, {int(index_values[i + 1])}, {int(index_values[i + 2])})")
    return mesh_var


def emit_infinite_light(
    emitter: Emitter,
    light: InfiniteLightDef,
    source_dir: Path,
    output_dir: Path,
    source_path: Path,
) -> None:
    filename = get_string(light.params, "filename")
    scale = get_float(light.params, "scale", 1.0)
    l_values = maybe_list(light.params.get("L"))

    if filename and l_values:
        fail("infinite light cannot specify both 'filename' and 'L'", source_path, light.line)
    if l_values:
        fail("constant infinite lights using 'L' are not supported by this converter", source_path, light.line)
    if not filename:
        fail("infinite light is missing 'filename'", source_path, light.line)

    envmap_path = resolve_asset_path(filename, source_dir, output_dir)
    envmap_var = emitter.unique("envMap")
    emitter.add(f"{envmap_var} = EnvMap({envmap_path!r})")
    if abs(scale - 1.0) > 1e-9:
        emitter.add(f"{envmap_var}.intensity = {format_number(scale)}")

    rotation = envmap_rotation_from_transform(light.transform)
    if not is_close_vec(rotation, [0.0, 0.0, 0.0], eps=1e-4):
        emitter.add(f"{envmap_var}.rotation = {format_float3(rotation)}")
    emitter.add(f"sceneBuilder.envMap = {envmap_var}")
    emitter.add()


def emit_scene(scene: SceneData, source_path: Path, output_path: Path, use_pbrt_materials: bool) -> str:
    emitter = Emitter()
    source_dir = source_path.parent
    output_dir = output_path.parent

    emitter.add("############################################################################")
    emitter.add("# Auto-generated by scripts/pbrt2pyscene.py")
    emitter.add(f"# Source: {source_path.as_posix()}")
    emitter.add("# Supported subset: Transform / Camera / Texture(imagemap) / MakeNamedMaterial /")
    emitter.add("# NamedMaterial / AreaLightSource(diffuse) / LightSource(infinite) /")
    emitter.add("# Shape(plymesh, trianglemesh)")
    emitter.add("############################################################################")
    emitter.add()

    if scene.integrator_name:
        emitter.add(f"# Integrator: {scene.integrator_name}")
    if scene.sampler_name:
        emitter.add(f"# Sampler: {scene.sampler_name}")
    if scene.pixel_filter_name:
        emitter.add(f"# PixelFilter: {scene.pixel_filter_name}")
    if scene.film_params:
        xres = get_float(scene.film_params, "xresolution", 0.0)
        yres = get_float(scene.film_params, "yresolution", 0.0)
        filename = get_string(scene.film_params, "filename", "")
        if xres and yres:
            emitter.add(f"# Film: {int(xres)}x{int(yres)}")
        if filename:
            emitter.add(f"# Film filename: {filename}")
        emitter.add()

    position, target, up = convert_camera_transform(scene.camera_transform)
    focal_length = fov_y_to_focal_length_mm(get_float(scene.camera_params, "fov", 90.0))
    focal_distance = get_float(scene.camera_params, "focaldistance", 1.0e30)
    aperture_radius = get_float(scene.camera_params, "lensradius", 0.0)

    emitter.add("camera = Camera()")
    emitter.add(f"camera.position = {format_float3(position)}")
    emitter.add(f"camera.target = {format_float3(target)}")
    emitter.add(f"camera.up = {format_float3(up)}")
    emitter.add(f"camera.focalLength = {format_number(focal_length)}")
    if focal_distance < 1.0e20:
        emitter.add(f"camera.focalDistance = {format_number(focal_distance)}")
    if aperture_radius != 0.0:
        emitter.add(f"camera.apertureRadius = {format_number(aperture_radius)}")
    emitter.add("sceneBuilder.addCamera(camera)")
    emitter.add()

    if scene.infinite_lights:
        if len(scene.infinite_lights) > 1:
            emitter.add(f"# Warning: {len(scene.infinite_lights)} infinite lights found; only the first is emitted.")
        emit_infinite_light(emitter, scene.infinite_lights[0], source_dir, output_dir, source_path)

    material_vars: Dict[str, str] = {}
    for material_name in scene.materials:
        material_vars[material_name] = emit_material(
            emitter, scene.materials[material_name], scene.textures, source_dir, output_dir, source_path, use_pbrt_materials
        )
        emitter.add()

    default_material_var = emitter.unique("DefaultMaterial")
    emitter.add(f"{default_material_var} = StandardMaterial('Default')")
    emitter.add(f"{default_material_var}.baseColor = float4(0.5, 0.5, 0.5, 1.0)")
    emitter.add(f"{default_material_var}.roughness = 1.0")
    emitter.add(f"{default_material_var}.doubleSided = True")
    emitter.add()

    for shape in scene.shapes:
        if shape.area_light is not None:
            material_var = emit_area_light_material(emitter, shape, shape.material_name, source_path)
        elif shape.material_name and shape.material_name in material_vars:
            material_var = material_vars[shape.material_name]
        else:
            material_var = default_material_var

        if shape.shape_type == "plymesh":
            filename = resolve_asset_path(get_string(shape.params, "filename"), source_dir, output_dir)
            mesh_name = Path(filename).stem
            mesh_var = emitter.unique(mesh_name)
            emitter.add(f"{mesh_var} = TriangleMesh.createFromFile({filename!r})")
        else:
            mesh_var = emit_trianglemesh_inline(emitter, shape, source_path)

        mesh_id_var = emitter.unique(f"mesh_{shape.index}")
        node_id_var = emitter.unique(f"node_{shape.index}")
        node_name = get_string(shape.params, "filename", f"{shape.shape_type}_{shape.index}")
        node_name = Path(node_name).stem

        if shape.shape_type == "plymesh":
            emitter.add(f"if {mesh_var} is not None:")
            emitter.add(f"    {mesh_id_var} = sceneBuilder.addTriangleMesh({mesh_var}, {material_var})")
            emitter.add(f"    {node_id_var} = sceneBuilder.addNode({node_name!r}, {format_transform_expr(shape.transform)})")
            emitter.add(f"    sceneBuilder.addMeshInstance({node_id_var}, {mesh_id_var})")
            emitter.add("else:")
            emitter.add(f"    print(\"Warning: Skipping mesh '{node_name}' because '{filename}' failed to load.\")")
        else:
            emitter.add(f"{mesh_id_var} = sceneBuilder.addTriangleMesh({mesh_var}, {material_var})")
            emitter.add(f"{node_id_var} = sceneBuilder.addNode({node_name!r}, {format_transform_expr(shape.transform)})")
            emitter.add(f"sceneBuilder.addMeshInstance({node_id_var}, {mesh_id_var})")
        emitter.add()

    return emitter.text()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a subset of PBRT-v4 scenes to Falcor .pyscene")
    parser.add_argument("input", type=Path, help="Input .pbrt file")
    parser.add_argument("-o", "--output", type=Path, help="Output .pyscene file")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it already exists")
    parser.add_argument(
        "--use-pbrt-materials",
        action="store_true",
        help="Emit supported materials as PBRT*Material instead of converting them to StandardMaterial",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = args.input if args.input.is_absolute() else (Path.cwd() / args.input)
    if not input_path.is_file():
        fail(f"input file does not exist: {input_path}")
    if input_path.suffix.lower() != ".pbrt":
        fail("input must be a .pbrt file")

    if args.output:
        output_path = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    else:
        output_path = input_path.with_suffix(".pyscene")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        fail(f"output file already exists: {output_path} (pass --force to overwrite)")

    scene = parse_scene(input_path)
    output_text = emit_scene(scene, input_path, output_path, args.use_pbrt_materials)
    output_path.write_text(output_text, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
