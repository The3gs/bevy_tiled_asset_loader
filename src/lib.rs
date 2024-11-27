use std::future::Future;

use bevy::app::Plugin;
use bevy::asset::io::Reader;
use bevy::asset::Asset;
use bevy::asset::AssetApp;
use bevy::asset::AssetLoader;
use bevy::asset::AsyncReadExt;
use bevy::asset::Handle;

use bevy::asset::ParseAssetPathError;
use bevy::color::Srgba;
use bevy::math::IVec2;
use bevy::math::UVec2;
use bevy::math::Vec2;
use bevy::reflect::TypePath;
use bevy::render::texture::Image;

use bevy::sprite::TextureAtlasLayout;

use bevy::utils::ConditionalSendFuture;
use bevy::utils::HashMap;
use thiserror::Error;

use xmltree::Element;

use bevy::tasks::futures_lite::AsyncRead;

use async_recursion::async_recursion;

trait ElementUtil {
    fn get_required_attr<T>(
        &self,
        s: &'static str,
        f: impl FnOnce(&String) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<T, TiledAssetLoaderError>;

    fn get_attr_or_default<T: Default>(
        &self,
        s: &str,
        f: impl FnOnce(&String) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<T, TiledAssetLoaderError>;

    fn get_optional_attr<T: Default>(
        &self,
        s: &str,
        f: impl FnOnce(&String) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<Option<T>, TiledAssetLoaderError>;

    fn get_single_child<T>(
        &self,
        s: &'static str,
        f: impl FnOnce(&Element) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<T, TiledAssetLoaderError>;

    fn get_children_with_name<'a>(&'a self, s: &'a str) -> impl Iterator<Item = &'a Element>;
}

impl ElementUtil for Element {
    fn get_required_attr<T>(
        &self,
        s: &'static str,
        f: impl FnOnce(&String) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<T, TiledAssetLoaderError> {
        f(self
            .attributes
            .get(s)
            .ok_or(TiledAssetLoaderError::MissingField(s))?)
    }

    fn get_attr_or_default<T: Default>(
        &self,
        s: &str,
        f: impl FnOnce(&String) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<T, TiledAssetLoaderError> {
        Ok(self
            .attributes
            .get(s)
            .map(f)
            .transpose()?
            .unwrap_or_default())
    }

    fn get_optional_attr<T: Default>(
        &self,
        s: &str,
        f: impl FnOnce(&String) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<Option<T>, TiledAssetLoaderError> {
        self.attributes.get(s).map(f).transpose()
    }

    fn get_single_child<T>(
        &self,
        s: &'static str,
        f: impl FnOnce(&Element) -> Result<T, TiledAssetLoaderError>,
    ) -> Result<T, TiledAssetLoaderError> {
        f(self
            .get_child(s)
            .ok_or(TiledAssetLoaderError::MissingField(s))?)
    }

    fn get_children_with_name<'a>(&'a self, s: &'a str) -> impl Iterator<Item = &'a Element> {
        self.children.iter().filter_map(move |e| match e {
            xmltree::XMLNode::Element(e) => (e.name == s).then_some(e),
            _ => None,
        })
    }
}

pub struct TiledMapPlugin;

impl Plugin for TiledMapPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_asset_loader::<TiledMapLoader>();
        app.init_asset_loader::<TiledSetLoader>();

        app.init_asset::<TiledMap>();
        app.init_asset::<TiledSet>();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId(usize);

#[derive(Debug, Clone, Copy)]
pub enum StaggerAxis {
    X,
    Y,
}

#[derive(Debug, Clone, Copy)]
pub enum StaggerIndex {
    Even,
    Odd,
}

#[derive(Debug, Clone)]
pub enum MapOrientation {
    Orthogonal,
    Isometric,
    Staggered(StaggerAxis, StaggerIndex),
    Hexagonal {
        side_length: usize,
        axis: StaggerAxis,
        index: StaggerIndex,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub enum MapRenderOrder {
    #[default]
    RightDown,
    RightUp,
    LeftDown,
    LeftUp,
}

#[derive(Debug, Clone, Default)]
pub struct EditorSettings {
    pub chunk_size: Option<UVec2>,
    pub export: Option<(String, String)>,
}

fn parse_editor_settings(e: &Element) -> Result<EditorSettings, TiledAssetLoaderError> {
    Ok(EditorSettings {
        chunk_size: todo!(),
        export: todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub class: String,
    pub data: LayerData,
    pub properties: HashMap<String, Property>,
    pub offset: IVec2,
    pub opacity: f32,
    pub parallax: Vec2,
    pub tint_color: Option<Srgba>,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum ObjectGroupDrawOrder {
    #[default]
    TopDown,
    Index,
}

#[derive(Debug, Clone)]
pub enum LayerData {
    Group {
        children: HashMap<LayerId, Layer>,
    },
    Layer {
        size: UVec2,
        data: Vec<Vec<usize>>,
    },
    ObjectGroup {
        color: Option<Srgba>,
        draw_order: ObjectGroupDrawOrder,
        objects: Vec<Object>,
    },
    ImageLayer {
        repeat_x: bool,
        repeat_y: bool,
        image: Handle<Image>,
    },
}

#[derive(Debug, Clone, TypePath, Asset)]
pub struct TiledMap {
    pub class: String,
    pub orientation: MapOrientation,
    pub render_order: MapRenderOrder,
    pub map_size: UVec2,
    pub tile_size: Vec2,
    pub paralax_origin: IVec2,
    pub background_color: Option<Srgba>,
    pub next_layer_id: usize,
    pub next_object_id: usize,
    pub infinite: bool,
    pub properties: HashMap<String, Property>,
    pub editor_settings: EditorSettings,
    pub tile_sets: Vec<(usize, Handle<TiledSet>)>,
    pub layers: Vec<(LayerId, Layer)>,
}

#[derive(Default)]
pub struct TiledMapLoader;

#[derive(Debug, Error)]
pub enum TiledAssetLoaderError {
    #[error("Could not load File")]
    Io(#[from] std::io::Error),
    #[error("Invalid UTF8")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("Error parsing {0:?}")]
    XmlParseError(#[from] xmltree::ParseError),
    #[error("Invalid number: {0}")]
    Number(#[from] std::num::ParseIntError),
    #[error("Invalid field {0}")]
    InvalidValue(String),
    #[error("Invalid Argument Name {0}")]
    InvalidArgument(String),
    #[error("Missing Required Field {0}")]
    MissingField(&'static str),
    #[error("Tag not allowed within parent")]
    InvalidTag(String),
    #[error("Invlaid floating point number")]
    InvalidFloat(#[from] std::num::ParseFloatError),
    #[error("Unxpected Event while parsing xml")]
    UnexpectedEvent,
    #[error(
        "File uses an unsupported feature (Usually depricated things I can't be bothered with)"
    )]
    UnsupportedFeature,
    #[error("Expected csv encoded layer data.")]
    ExpectedCsvData,
    #[error("Invalid Path for needed asset {0:?}")]
    ParsePath(#[from] ParseAssetPathError),
}

impl AssetLoader for TiledMapLoader {
    type Asset = TiledMap;
    type Settings = ();
    type Error = TiledAssetLoaderError;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _setting: &'a Self::Settings,
        load_context: &'a mut bevy::asset::LoadContext,
    ) -> impl ConditionalSendFuture
           + Future<Output = Result<<Self as AssetLoader>::Asset, <Self as AssetLoader>::Error>>
    {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;

            let set = Element::parse(bytes.as_slice())?;

            parse_tilemap(set, load_context).await
        })
    }

    fn extensions(&self) -> &[&str] {
        &["tmx"]
    }
}

async fn parse_tilemap(
    map: Element,
    load_context: &mut bevy::asset::LoadContext<'_>,
) -> Result<TiledMap, TiledAssetLoaderError> {
    let class = map.get_attr_or_default("class", |s| Ok(s.clone()))?;

    let orientation = map.get_required_attr("orientation", |s| match s.as_str() {
        "orthogonal" => Ok(MapOrientation::Orthogonal),
        "staggered" => {
            let axis = map.get_required_attr("staggeraxis", |s| match s.as_str() {
                "x" => Ok(StaggerAxis::X),
                "y" => Ok(StaggerAxis::Y),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?;

            let index = map.get_required_attr("staggerindex", |s| match s.as_str() {
                "even" => Ok(StaggerIndex::Even),
                "odd" => Ok(StaggerIndex::Odd),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?;

            Ok(MapOrientation::Staggered(axis, index))
        }
        "isometric" => Ok(MapOrientation::Isometric),
        "hexagonal" => {
            let side_length = map.get_required_attr("hexsidelength", |s| Ok(s.parse()?))?;

            let axis = map.get_required_attr("staggeraxis", |s| match s.as_str() {
                "x" => Ok(StaggerAxis::X),
                "y" => Ok(StaggerAxis::Y),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?;

            let index = map.get_required_attr("staggerindex", |s| match s.as_str() {
                "even" => Ok(StaggerIndex::Even),
                "odd" => Ok(StaggerIndex::Odd),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?;

            Ok(MapOrientation::Hexagonal {
                side_length,
                axis,
                index,
            })
        }
        _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
    })?;

    let render_order = map.get_attr_or_default("renderorder", |s| match s.as_str() {
        "right-down" => Ok(MapRenderOrder::RightDown),
        "right-up" => Ok(MapRenderOrder::RightUp),
        "left-down" => Ok(MapRenderOrder::LeftDown),
        "left-up" => Ok(MapRenderOrder::LeftUp),
        _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
    })?;

    let width = map.get_required_attr("width", |s| Ok(s.parse::<u32>()?))?;
    let height = map.get_required_attr("height", |s| Ok(s.parse::<u32>()?))?;
    let tilewidth = map.get_required_attr("tilewidth", |s| Ok(s.parse::<f32>()?))?;
    let tileheight = map.get_required_attr("tileheight", |s| Ok(s.parse::<f32>()?))?;
    let parallaxoriginx = map.get_attr_or_default("parallaxoriginx", |s| Ok(s.parse::<i32>()?))?;
    let parallaxoriginy = map.get_attr_or_default("parallaxoriginy", |s| Ok(s.parse::<i32>()?))?;

    let background_color = map.get_optional_attr("backgroundcolor", parse_color)?;

    let next_layer_id = map.get_optional_attr("nextlayerid", |s| Ok(s.parse::<usize>()?))?;
    let next_object_id = map.get_optional_attr("nextobjectid", |s| Ok(s.parse::<usize>()?))?;

    let infinite = map.get_attr_or_default("infinite", |s| match s.as_str() {
        "1" => Ok(true),
        "0" => Ok(false),
        _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
    })?;

    let properties = map
        .get_child("properties")
        .map(parse_properties)
        .transpose()?
        .unwrap_or(HashMap::new());

    let editor_settings = map
        .get_child("editorsettings")
        .map(parse_editor_settings)
        .transpose()?
        .unwrap_or(EditorSettings::default());

    let mut tile_sets = Vec::new();
    for e in map.children.iter() {
        match e {
            xmltree::XMLNode::Element(e) => {
                if e.name == "tileset" {
                    let name = e.get_attr_or_default("name", |s| Ok(s.clone()))?;
                    let firstgid = e.get_required_attr("firstgid", |s| Ok(s.parse::<usize>()?))?;

                    let label = format!("{name}.{firstgid}");

                    let tile_set_source = e.attributes.get("source");
                    let tile_set = if let Some(source) = tile_set_source {
                        load_context.load(
                            load_context
                                .asset_path()
                                .parent()
                                .unwrap()
                                .resolve(source)?,
                        )
                    } else {
                        let tile_set = parse_tileset(e, load_context).await?;
                        load_context.add_labeled_asset(label, tile_set)
                    };
                    tile_sets.push((firstgid, tile_set));
                }
            }
            _ => (),
        }
    }

    let mut layers = Vec::new();
    for e in map.children.iter() {
        match e {
            xmltree::XMLNode::Element(e) => {
                if e.name == "layer"
                    || e.name == "group"
                    || e.name == "objectgroup"
                    || e.name == "imagelayer"
                {
                    let layer = parse_layer(e, load_context).await?;
                    layers.push((layer.0, layer.1));
                }
            }
            _ => (),
        }
    }

    Ok(TiledMap {
        class,
        orientation,
        render_order,
        map_size: UVec2::new(width, height),
        tile_size: Vec2::new(tilewidth, tileheight),
        paralax_origin: IVec2::new(parallaxoriginx, parallaxoriginy),
        background_color,
        next_layer_id: next_layer_id
            .unwrap_or_else(|| layers.iter().fold(1, |acc, (i, _)| (i.0 + 1).max(acc))),
        next_object_id: next_object_id.unwrap(),
        infinite,
        properties,
        editor_settings,
        tile_sets,
        layers,
    })
}

#[derive(Clone, Debug)]
pub struct Object {
    pub name: String,
    pub object_type: String,
    pub pos: Vec2,
    pub size: Vec2,
    pub rotation: f32,
    pub visible: bool,
    pub properties: HashMap<String, Property>,
}

fn parse_object(e: &Element) -> Result<Object, TiledAssetLoaderError> {
    let name = e.get_attr_or_default("name", |s| Ok(s.clone()))?;
    let object_type = e.get_attr_or_default("type", |s| Ok(s.clone()))?;
    let x = e.get_attr_or_default("x", |s| Ok(s.parse()?))?;
    let y = e.get_attr_or_default("y", |s| Ok(s.parse()?))?;
    let width = e.get_attr_or_default("width", |s| Ok(s.parse()?))?;
    let height = e.get_attr_or_default("height", |s| Ok(s.parse()?))?;
    let rotation = e.get_attr_or_default("rotation", |s| Ok(s.parse()?))?;
    let visible = e
        .get_optional_attr("visible", |s| match s.as_str() {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
        })?
        .unwrap_or(true);
    let properties = e
        .get_child("properties")
        .map(parse_properties)
        .transpose()?
        .unwrap_or_else(|| HashMap::new());
    Ok(Object {
        name,
        object_type,
        pos: Vec2::new(x, y),
        size: Vec2::new(width, height),
        rotation,
        visible,
        properties,
    })
}

#[async_recursion]
async fn parse_layer(
    e: &Element,
    load_context: &mut bevy::asset::LoadContext<'_>,
) -> Result<(LayerId, Layer), TiledAssetLoaderError> {
    let id = e.get_required_attr("id", |s| Ok(LayerId(s.parse()?)))?;

    let name = e.get_attr_or_default("name", |s| Ok(s.clone()))?;
    let class = e.get_attr_or_default("class", |s| Ok(s.clone()))?;

    let data = match e.name.as_str() {
        "layer" => LayerData::Layer {
            size: UVec2::new(
                e.get_required_attr("width", |s| Ok(s.parse()?))?,
                e.get_required_attr("height", |s| Ok(s.parse()?))?,
            ),
            data: e.get_single_child("data", parse_layer_data)?,
        },
        "group" => LayerData::Group {
            children: {
                let mut children = HashMap::new();
                for child in &e.children {
                    match child {
                        xmltree::XMLNode::Element(e)
                            if e.name == "layer"
                                || e.name == "group"
                                || e.name == "objectgroup"
                                || e.name == "imagelayer" =>
                        {
                            let layer = parse_layer(&e, load_context).await?;
                            children.insert(layer.0, layer.1);
                        }
                        _ => (),
                    }
                }
                children
            },
        },
        "objectgroup" => LayerData::ObjectGroup {
            color: e.get_optional_attr("color", parse_color)?,
            draw_order: e.get_attr_or_default("draworder", |s| match s.as_str() {
                "topdown" => Ok(ObjectGroupDrawOrder::TopDown),
                "index" => Ok(ObjectGroupDrawOrder::Index),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?,
            objects: e
                .get_children_with_name("object")
                .map(parse_object)
                .collect::<Result<Vec<_>, _>>()?,
        },
        "imagelayer" => LayerData::ImageLayer {
            repeat_x: e.get_attr_or_default("repeatx", |s| match s.as_str() {
                "0" => Ok(false),
                "1" => Ok(true),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?,
            repeat_y: e.get_attr_or_default("repeaty", |s| match s.as_str() {
                "0" => Ok(false),
                "1" => Ok(true),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?,
            image: {
                let image = e
                    .get_child("image")
                    .ok_or(TiledAssetLoaderError::MissingField("image"))?;

                parse_image(image, load_context).await?
            },
        },
        _ => return Err(TiledAssetLoaderError::InvalidValue(e.name.clone())),
    };

    let properties = e
        .get_child("properties")
        .map(parse_properties)
        .transpose()?
        .unwrap_or_else(|| HashMap::new());

    let offset_x = e.get_attr_or_default("offsetx", |s| Ok(s.parse()?))?;
    let offset_y = e.get_attr_or_default("offsety", |s| Ok(s.parse()?))?;

    let opacity = e
        .get_optional_attr("opacity", |s| Ok(s.parse()?))?
        .unwrap_or(1.0);

    let parallax_x = e.get_attr_or_default("parallaxoriginx", |s| Ok(s.parse()?))?;
    let parallax_y = e.get_attr_or_default("parallaxoriginy", |s| Ok(s.parse()?))?;

    let tint_color = e.get_optional_attr("tintcolor", parse_color)?;

    let visible = e
        .get_optional_attr("visible", |s| match s.as_str() {
            "0" => Ok(false),
            "1" => Ok(true),
            _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
        })?
        .unwrap_or(true);

    Ok((
        id,
        Layer {
            name,
            class,
            data,
            properties,
            offset: IVec2::new(offset_x, offset_y),
            opacity,
            parallax: Vec2::new(parallax_x, parallax_y),
            tint_color,
            visible,
        },
    ))
}

fn parse_layer_data(e: &Element) -> Result<Vec<Vec<usize>>, TiledAssetLoaderError> {
    let encoding = e.get_required_attr("encoding", |s| Ok(s.clone()))?;
    match encoding.as_str() {
        "csv" => match &e.children[..] {
            [xmltree::XMLNode::Text(data)] => data
                .split('\n')
                .filter(|l| l.trim().len() > 0)
                .map(|line| {
                    line.split(',')
                        .filter_map(|cell| {
                            let cell = cell.trim();
                            if cell.is_empty() {
                                None
                            } else {
                                Some(
                                    cell.trim()
                                        .parse::<usize>()
                                        .map_err(TiledAssetLoaderError::Number),
                                )
                            }
                        })
                        .collect()
                })
                .collect(),
            _ => Err(TiledAssetLoaderError::ExpectedCsvData),
        },
        _ => return Err(TiledAssetLoaderError::InvalidValue(encoding.clone())),
    }
}

fn parse_color(s: &String) -> Result<Srgba, TiledAssetLoaderError> {
    let code = s.as_str();
    let code = if code.starts_with('#') {
        &code[1..]
    } else {
        code
    };

    fn from_hex(b: &u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b + 10 - b'a'),
            b'A'..=b'F' => Some(b + 10 - b'A'),
            _ => None,
        }
    }

    fn hex_byte(b0: &u8, b1: &u8) -> Option<u8> {
        Some(from_hex(b0)? * 16 + from_hex(b1)?)
    }

    match code.as_bytes() {
        [a0, a1, r0, r1, g0, g1, b0, b1] => Ok(Srgba::rgba_u8(
            hex_byte(r0, r1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
            hex_byte(g0, g1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
            hex_byte(b0, b1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
            hex_byte(a0, a1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
        )),
        [r0, r1, g0, g1, b0, b1] => Ok(Srgba::rgb_u8(
            hex_byte(r0, r1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
            hex_byte(g0, g1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
            hex_byte(b0, b1).ok_or_else(|| TiledAssetLoaderError::InvalidValue(s.clone()))?,
        )),
        _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
    }
}

#[derive(Debug, Clone)]
pub struct AnimationFrame {
    pub tile_id: usize,
    pub duration: f64,
}

#[derive(Debug, Clone)]
pub struct TileData {
    pub tile_type: String,
    pub probability: f32,
    pub pos: UVec2,
    pub dimensions: UVec2,
    pub image: Option<Handle<Image>>,
    pub animation: Option<Vec<AnimationFrame>>,
    pub properties: HashMap<String, Property>,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum TileObjectAlignment {
    #[default]
    Unspecified,
    TopLeft,
    Top,
    TopRight,
    Left,
    Center,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum TileSetFillMode {
    #[default]
    Stretch,
    PreserveAspectFit,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum RenderMode {
    #[default]
    Tile,
    Grid,
}

#[derive(TypePath, Clone, Asset, Debug)]
pub struct TiledSet {
    pub name: String,
    pub class: String,
    pub tile_count: u32,
    pub columns: u32,
    pub object_alignment: TileObjectAlignment,
    pub render_mode: RenderMode,
    pub fill_mode: TileSetFillMode,
    pub texture: Option<(Handle<TextureAtlasLayout>, Handle<Image>)>,
    pub tiles: HashMap<usize, TileData>,
    pub properties: HashMap<String, Property>,
}

#[derive(Debug, Clone)]
pub enum Property {
    String(String),
    Int(i64),
    Bool(bool),
    Float(f32),
    Color([u8; 4]),
    File(String),
    Object(usize),
}

#[derive(Default)]
pub struct TiledSetLoader;

impl AssetLoader for TiledSetLoader {
    type Asset = TiledSet;
    type Settings = ();
    type Error = TiledAssetLoaderError;

    fn load<'a>(
        &'a self,
        reader: &'a mut Reader,
        _setting: &'a Self::Settings,
        load_context: &'a mut bevy::asset::LoadContext,
    ) -> impl ConditionalSendFuture
           + Future<Output = Result<<Self as AssetLoader>::Asset, <Self as AssetLoader>::Error>>
    {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;

            let set = Element::parse(bytes.as_slice())?;

            parse_tileset(&set, load_context).await
        })
    }

    fn extensions(&self) -> &[&str] {
        &["tsx"]
    }
}

async fn parse_tileset(
    set: &Element,
    load_context: &mut bevy::asset::LoadContext<'_>,
) -> Result<TiledSet, TiledAssetLoaderError> {
    if set.name == "tileset" {
        let name = set
            .attributes
            .get("name")
            .ok_or(TiledAssetLoaderError::MissingField("Name"))?
            .clone();

        let class = set.attributes.get("name").cloned().unwrap_or(String::new());

        let tilewidth = set.get_attr_or_default("tilewidth", |n| Ok(n.parse::<f32>()?))?;
        let tileheight = set.get_attr_or_default("tileheight", |n| Ok(n.parse::<f32>()?))?;
        let spacing = set.get_attr_or_default("spacing", |n| Ok(n.parse::<u32>()?))?;
        let margin = set.get_attr_or_default("margin", |n| Ok(n.parse::<u32>()?))?;
        let tilecount = set.get_attr_or_default("tilecount", |n| Ok(n.parse::<u32>()?))?;
        let columns = set.get_attr_or_default("columns", |n| Ok(n.parse::<u32>()?))?;

        let object_alignment =
            set.get_attr_or_default("objectalignment", |s| match s.as_str() {
                "unspecified" => Ok(TileObjectAlignment::Unspecified),
                "topleft" => Ok(TileObjectAlignment::TopLeft),
                "top" => Ok(TileObjectAlignment::Top),
                "topright" => Ok(TileObjectAlignment::TopRight),
                "left" => Ok(TileObjectAlignment::Left),
                "center" => Ok(TileObjectAlignment::Center),
                "right" => Ok(TileObjectAlignment::Right),
                "bottomleft" => Ok(TileObjectAlignment::BottomLeft),
                "bottom" => Ok(TileObjectAlignment::Bottom),
                "bottomright" => Ok(TileObjectAlignment::BottomRight),
                _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
            })?;

        let render_mode = set.get_attr_or_default("tilerendersize", |s| match s.as_str() {
            "tile" => Ok(RenderMode::Tile),
            "grid" => Ok(RenderMode::Grid),
            _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
        })?;

        let fill_mode = set.get_attr_or_default("fillmode", |s| match s.as_str() {
            "stretch" => Ok(TileSetFillMode::Stretch),
            "preserve-aspect-fit" => Ok(TileSetFillMode::PreserveAspectFit),
            _ => Err(TiledAssetLoaderError::InvalidValue(s.clone())),
        })?;

        let texture_layout_data = TextureAtlasLayout::from_grid(
            UVec2::new(tileheight as u32, tilewidth as u32),
            columns,
            tilecount / columns,
            Some(UVec2::new(spacing, spacing)),
            Some(UVec2::new(margin, margin)),
        );

        let layout_handle =
            load_context.add_labeled_asset(format!("{name}[layout]"), texture_layout_data);

        let image = set.get_child("image");

        let image: Option<Handle<Image>> = if let Some(image) = image {
            Some(parse_image(image, load_context).await?)
        } else {
            None
        };

        let mut tiles = HashMap::new();
        for e in set.children.iter() {
            match e {
                xmltree::XMLNode::Element(e) => {
                    if e.name == "tile" {
                        let tile = parse_tile(e, load_context).await?;
                        tiles.insert(tile.0, tile.1);
                    }
                }
                _ => (),
            }
        }

        let properties = set
            .get_child("properties")
            .map(parse_properties)
            .unwrap_or(Ok(HashMap::new()))?;

        Ok(TiledSet {
            name,
            class,
            tile_count: tilecount,
            columns,
            object_alignment,
            render_mode,
            fill_mode,
            texture: image.map(|image| (layout_handle, image)),
            tiles,
            properties,
        })
    } else {
        Err(TiledAssetLoaderError::InvalidTag(set.name.clone()))
    }
}

async fn parse_tile(
    e: &Element,
    load_context: &mut bevy::asset::LoadContext<'_>,
) -> Result<(usize, TileData), TiledAssetLoaderError> {
    let id = e
        .attributes
        .get("id")
        .ok_or(TiledAssetLoaderError::MissingField("id"))
        .map(|n| n.parse::<usize>())??;

    let tile_type = e
        .attributes
        .get("type")
        .or(e.attributes.get("class"))
        .cloned()
        .unwrap_or(String::new());

    let probability = e
        .attributes
        .get("probability")
        .map(|n| n.parse::<f32>())
        .unwrap_or(Ok(0.0))?;

    let x = e
        .attributes
        .get("x")
        .map(|n| n.parse::<u32>())
        .unwrap_or(Ok(0))?;
    let y = e
        .attributes
        .get("y")
        .map(|n| n.parse::<u32>())
        .unwrap_or(Ok(0))?;
    let width = e
        .attributes
        .get("width")
        .map(|n| n.parse::<u32>())
        .unwrap_or(Ok(0))?;
    let height = e
        .attributes
        .get("height")
        .map(|n| n.parse::<u32>())
        .unwrap_or(Ok(0))?;

    let properties = e
        .get_child("properties")
        .map(parse_properties)
        .unwrap_or(Ok(HashMap::new()))?;

    let image = match e.get_child("image") {
        Some(image) => Some(parse_image(image, load_context).await?),
        None => None,
    };

    Ok((
        id,
        TileData {
            tile_type,
            probability,
            pos: UVec2::new(x, y),
            dimensions: UVec2::new(width, height),
            image,
            animation: None,
            properties,
        },
    ))
}

#[derive(Debug, Clone)]
struct Base64Reader<'a> {
    data: &'a [u8],
    count: u8,
    residual: u16,
}

fn base_64_bit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'z' => Some(byte - b'a' + (b'9' - b'0' + 1)),
        b'A'..=b'Z' => Some(byte - b'A' + (b'z' - b'a' + 1) + (b'9' - b'0' + 1)),
        _ => None,
    }
}

impl<'a> AsyncRead for Base64Reader<'a> {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let s = std::pin::Pin::into_inner(self);
        for (i, cell) in buf.iter_mut().enumerate() {
            while s.count < 8 {
                if s.data.len() == 0 {
                    return std::task::Poll::Ready(Ok(i));
                }
                s.residual |= (s.data[0] as u16) << s.count;
                s.data = &s.data[1..];
            }

            *cell = (s.residual & 0xff) as u8;
            s.count -= 8;
            s.residual >>= 8;
        }

        std::task::Poll::Ready(Ok(buf.len()))
    }
}

async fn parse_image(
    e: &Element,
    load_context: &mut bevy::asset::LoadContext<'_>,
) -> Result<Handle<Image>, TiledAssetLoaderError> {
    if let Some(source) = e.attributes.get("source") {
        // the image is not embedded. load normally
        Ok(load_context.load(
            load_context
                .asset_path()
                .parent()
                .unwrap()
                .resolve(source)?,
        ))
    } else {
        let _format = e
            .attributes
            .get("format")
            .ok_or(TiledAssetLoaderError::MissingField("source or format"))?;

        todo!("Implement embedded images")
    }
}

fn parse_properties(e: &Element) -> Result<HashMap<String, Property>, TiledAssetLoaderError> {
    e.children
        .iter()
        .map(|e| match e {
            xmltree::XMLNode::Element(prop) => {
                if prop.name == "property" {
                    let name = prop
                        .attributes
                        .get("name")
                        .ok_or(TiledAssetLoaderError::MissingField("name"))?
                        .clone();
                    let typ = prop
                        .attributes
                        .get("type")
                        .cloned()
                        .unwrap_or(String::from("string"));
                    // let custom_type = prop.attributes.get("propetytype").cloned();
                    let value = prop.attributes.get("value");

                    Ok((
                        name,
                        match typ.as_str() {
                            "string" => {
                                Property::String(value.map(String::from).unwrap_or_default())
                            }
                            "int" => Property::Int(value.map(|v| v.parse()).unwrap_or(Ok(0))?),
                            "bool" => Property::Bool(value.is_none() || value.unwrap() == "true"),
                            "float" => {
                                Property::Float(value.map(|v| v.parse()).unwrap_or(Ok(0.0))?)
                            }
                            _ => return Err(TiledAssetLoaderError::InvalidValue(typ)),
                        },
                    ))
                } else {
                    Err(TiledAssetLoaderError::InvalidArgument(prop.name.clone()))
                }
            }
            _ => Err(TiledAssetLoaderError::InvalidArgument(String::new())),
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn tileset_file() {
        use bevy::prelude::*;

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            ImagePlugin::default(),
        ));
        app.init_asset_loader::<TiledSetLoader>();
        app.init_asset::<TiledSet>();

        let asset_server = app.world().resource::<AssetServer>();

        let handle: Handle<TiledSet> = asset_server.load("NonEmbedded.tsx");

        for _ in 0..1000 {
            app.update();
        }

        app.world()
            .resource::<Assets<TiledSet>>()
            .get(&handle)
            .unwrap();
    }

    #[test]
    fn nonembedded_tilemap() {
        use bevy::prelude::*;

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            ImagePlugin::default(),
        ));

        app.init_asset_loader::<TiledSetLoader>();
        app.init_asset_loader::<TiledMapLoader>();
        app.init_asset::<TiledSet>();
        app.init_asset::<TiledMap>();

        let asset_server = app.world().resource::<AssetServer>();

        let handle: Handle<TiledMap> = asset_server.load("NonEmbeddedTilemap.tmx");

        for _ in 0..100 {
            app.update();
        }

        let map = app
            .world()
            .resource::<Assets<TiledMap>>()
            .get(&handle)
            .unwrap();

        assert_eq!(map.layers.len(), 1);
        let (
            id,
            Layer {
                name,
                class,
                data,
                properties,
                offset,
                opacity,
                parallax,
                tint_color,
                visible,
            },
        ) = map.layers.iter().next().unwrap();

        assert_eq!(id.0, 1);
        assert_eq!(name.as_str(), "Tile Layer 1");
        assert_eq!(class.as_str(), "");
        match data {
            LayerData::Group { .. } => panic!(),
            LayerData::Layer { size, data } => {
                println!("{data:?}");
                assert_eq!(size.y as usize, data.len());
                assert_eq!(size.x as usize, data[0].len());
            }
            LayerData::ObjectGroup { .. } => panic!(),
            LayerData::ImageLayer { .. } => panic!(),
        }
        assert!(properties.is_empty());
        assert_eq!(offset, &IVec2::new(0, 0));
        assert_eq!(opacity, &1.0);
        assert_eq!(parallax, &Vec2::new(0.0, 0.0));
        assert!(tint_color.is_none());
        assert!(visible);
    }
}
