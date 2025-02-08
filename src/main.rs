use bevy::prelude::*;
use clap::Parser;
use csv::Writer;
use polars::prelude::*;
use polars::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::Rng;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;

impl Distribution<Class> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Class {
        match rng.random_range(0..10) {
            0 => Class::Bourgeois,
            1 => Class::Bourgeois,
            _ => Class::Proletariat,
        }
    }
}

impl Distribution<Income> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Income {
        match rng.random_range(0..100) {
            0..=70 => Income::Low,
            71..=90 => Income::Middle,
            91..=98 => Income::High,
            99..=100 => Income::VeryHigh,
            _ => Income::Low,
        }
    }
}
#[derive(Serialize)]
enum Class {
    Bourgeois,
    Proletariat,
}

#[derive(Serialize)]
enum GoodsType {
    Food,
    Steel,
    Coal,
    Iron,
    Copper,
    Oil,
    Car,
    Tool,
    Weapon,
    Clothes,
    Furniture,
    Wine,
    Tobacco,
    Rent,
    Wood,
}

#[derive(Serialize)]
enum Income {
    Low,
    Middle,
    High,
    VeryHigh,
}

#[derive(Serialize)]
enum OrderType {
    Production,
    Person,
}


#[derive(Serialize)]
struct Person {
    id: i64,
    class: Class,
    income: Income,
    money: f64,
}

#[derive(Serialize)]
struct Production {
    id: i64,
    input: Vec<GoodsType>,
    output: GoodsType,
}

#[derive(Serialize)]
struct OwnerShip {
    id: i64,
    owner_id: i64,
    production_id: i64
}

struct BuyOrder {
    id: i64,
    person_or_production_id: i64,
    goods_type: GoodsType,
    amount: i64,
    price: f64,
    order_type: OrderType,
}

struct SellOrder {
    id: i64,
    person_or_production_id: i64,
    goods_type: GoodsType,
    amount: i64,
    price: f64,
    order_type: OrderType,
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {}

#[derive(Resource)]
struct DataContext {
    df: LazyFrame,
}

fn createCsv() {
    let mut vec = Vec::new();
    for i in 0..100 {
        let person = Person {
            id: i,
            class: rand::random(),
            income: rand::random(),
            money: rand::random::<f64>() * 10000.0,
        };
        vec.push(person);
    }
    let mut wtr = Writer::from_writer(vec![]);
    for person in vec {
        wtr.serialize(person).unwrap();
    }
    wtr.flush().unwrap();
    let csv_data = String::from_utf8(wtr.into_inner().unwrap_or_default()).unwrap_or_default();
    let mut file = File::create("output.csv").unwrap();
    file.write_all(csv_data.as_bytes()).unwrap();
}
fn main() {
    let _ = Cli::parse();

    createCsv();

    let file = std::fs::File::open("output.csv").expect("Could not open file");
    let _ = CsvReader::new(file)
        .finish()
        .expect("Could not create DataFrame");

    App::new()
        .add_plugins(MinimalPlugins)
        .add_systems(Startup, startup)
        .insert_resource(Time::<Fixed>::from_duration(Duration::from_millis(1000)))
        .add_systems(FixedUpdate, main_loop)
        .insert_resource(Time::<Fixed>::from_seconds(1.0))
        .run();
}

fn startup(mut commands: Commands) {
    let file = std::fs::File::open("output.csv").expect("Could not open file");
    let df = CsvReader::new(file)
        .finish()
        .expect("Could not create DataFrame");
    let df = df.lazy();
    commands.insert_resource(DataContext { df });
}

fn main_loop(data_context: Res<DataContext>) {
    let df = data_context.df.clone();
    println!("{}", &df.collect().unwrap());
}
