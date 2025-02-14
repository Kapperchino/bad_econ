use bevy::prelude::*;
use bevy::utils::HashMap;
use clap::Parser;
use csv::Writer;
use polars::prelude::*;
use polars::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::{rng, Rng};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use strum::IntoEnumIterator; // 0.17.1
use strum_macros::EnumIter; // 0.17.1

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
            _ => Income::Low,
        }
    }
}

#[derive(Serialize)]
pub enum Class {
    Bourgeois,
    Proletariat,
}

#[derive(Serialize, Component,EnumIter, PartialEq, Eq, Hash, Clone,Copy, Debug)]
pub enum GoodsType {
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
    Labor,
    Cotton,
}

impl GoodsType {
    pub fn starting_price(&self) -> f64 {
        match self {
            GoodsType::Food => 1.0,
            GoodsType::Steel => 1.0,
            GoodsType::Coal => 1.0,
            GoodsType::Iron => 1.0,
            GoodsType::Copper => 1.0,
            GoodsType::Oil => 1.0,
            GoodsType::Car => 1.0,
            GoodsType::Tool => 1.0,
            GoodsType::Weapon => 1.0,
            GoodsType::Clothes => 1.0,
            GoodsType::Furniture => 1.0,
            GoodsType::Wine => 1.0,
            GoodsType::Tobacco => 1.0,
            GoodsType::Rent => 1.0,
            GoodsType::Wood => 1.0,
            GoodsType::Labor => 1.0,
            GoodsType::Cotton => 1.0,
        }
    }
}

#[derive(Serialize, Clone)]
pub enum Income {
    Low,
    Middle,
    High,
    VeryHigh,
}

impl Income {
    pub fn starting_money(&self) -> f64 {
        match self {
            Income::Low => 10.0,
            Income::Middle => 50.0,
            Income::High => 100.0,
            Income::VeryHigh => 1500.0,
        }
    }
}
#[derive(Serialize, Clone)]
pub enum OrderType {
    Production,
    Person,
}

#[derive(Serialize, Component)]
pub struct Person {
    pub id: i64,
    pub class: Class,
    pub income: Income,
    pub money: f64,
}

#[derive(Serialize, Component, Debug)]
pub enum Production {
    SteelMill,
    IronMine,
    CoalMine,
    CopperMine,
    OilWell,
    CarFactory,
    ToolFactory,
    WeaponFactory,
    ClothesFactory,
    FurnitureFactory,
    WineFactory,
    TobaccoFarm,
    House,
    SawMill,
    CottonFarm,
    // add other production types here
}

impl Production {
    /// Returns the list of inputs for this production facility.
    pub fn input(&self) -> Vec<GoodsType> {
        match self {
            Production::SteelMill => vec![GoodsType::Labor, GoodsType::Iron, GoodsType::Coal],
            Production::IronMine => vec![GoodsType::Labor, GoodsType::Tool],
            Production::CoalMine => vec![GoodsType::Labor, GoodsType::Tool],
            Production::CopperMine => vec![GoodsType::Labor, GoodsType::Tool],
            Production::OilWell => vec![GoodsType::Labor, GoodsType::Tool],
            Production::CarFactory => vec![GoodsType::Labor, GoodsType::Steel],
            Production::ToolFactory => vec![GoodsType::Labor, GoodsType::Steel],
            Production::WeaponFactory => vec![GoodsType::Labor, GoodsType::Steel, GoodsType::Tool],
            Production::ClothesFactory => vec![GoodsType::Labor, GoodsType::Tool],
            Production::FurnitureFactory => vec![GoodsType::Labor,GoodsType::Wood, GoodsType::Tool],
            Production::WineFactory => vec![GoodsType::Labor,  GoodsType::Tool],
            Production::TobaccoFarm => vec![GoodsType::Labor],
            Production::House => vec![],
            Production::SawMill => vec![GoodsType::Labor,GoodsType::Tool],
            Production::CottonFarm => vec![GoodsType::Labor,GoodsType::Tool],
        }
    }

    /// Returns the output good for this production facility.
    pub fn output(&self) -> GoodsType {
        match self {
            Production::SteelMill => GoodsType::Steel,
            Production::IronMine => GoodsType::Iron,
            Production::CoalMine => GoodsType::Coal,
            Production::CopperMine => GoodsType::Copper,
            Production::OilWell => GoodsType::Oil,
            Production::CarFactory => GoodsType::Car,
            Production::ToolFactory => GoodsType::Tool,
            Production::WeaponFactory => GoodsType::Weapon,
            Production::ClothesFactory => GoodsType::Clothes,
            Production::FurnitureFactory => GoodsType::Furniture,
            Production::WineFactory => GoodsType::Wine,
            Production::TobaccoFarm => GoodsType::Tobacco,
            Production::House => GoodsType::Rent,
            Production::SawMill => GoodsType::Wood,
            Production::CottonFarm => GoodsType::Cotton,
        }
    }
}

pub struct ProductionInstance {
    id: i64,
    production: Production,
}

#[derive(Component)]
pub struct Price {
    pub goods_type: GoodsType,
    pub price: f64,
}

#[derive(Bundle)]
pub struct BuyOrders {
    pub goods_type: GoodsType,
    pub buy_order: BuyOrder,
}

#[derive(Bundle)]
pub struct SellOrders {
    pub goods_type: GoodsType,
    pub sell_order: SellOrder,
}

#[derive(Component, Clone)]
pub struct BuyOrder {
    pub goods_type: GoodsType,
    pub amount: i64,
    pub price: f64,
    pub order_type: OrderType,
}

#[derive(Clone,Copy, Debug)]
pub struct BuyOrderSum {
    pub goods_type: GoodsType,
    pub amount: i64,
    pub price: f64,
}

#[derive(Component, Clone)]
pub struct SellOrder {
    goods_type: GoodsType,
    amount: i64,
    price: f64,
    order_type: OrderType,
}