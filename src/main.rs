mod models;

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

use models::*;

#[derive(Parser)]
#[command(version, about)]
struct Cli {}


fn main() {
    let _ = Cli::parse();

    App::new()
        .add_plugins(MinimalPlugins)
        .add_systems(Startup, startup)
        .insert_resource(Time::<Fixed>::from_seconds(1.0))
        .add_systems(FixedUpdate, (main_loop, market_loop))
        .run();
}

fn startup(mut commands: Commands) {
    for i in 0..100000 {
        let class : Class = rng().random();
        let income : Income = rng().random();
        match class {
            Class::Bourgeois => {
                commands.spawn(Person {
                    id: i,
                    class: class,
                    income: Income::VeryHigh,
                    money: Income::VeryHigh.starting_money(),
                });
            },
            Class::Proletariat => {
                commands.spawn(Person {
                    id: i,
                    class: class,
                    income: income.clone(),
                    money: income.starting_money(),
                });
            },
        }
    }
    for goods_type in GoodsType::iter() {
        let price = goods_type.starting_price().clone();
        commands.spawn(Price {
            goods_type,
            price,
        });
    };
}

fn main_loop(
    people_query: Query<&Person>,
    prices_query: Query<&Price>,
    mut commands: Commands,
) {
    for person in people_query.iter() {
        match person.class {
            Class::Bourgeois => {
                // TODO: Implement bourgeois behavior
            },
            Class::Proletariat => {
                let food_price = prices_query.iter()
                    .find(|p| p.goods_type == GoodsType::Food)
                    .unwrap()
                    .price;
                match person.income {
                    Income::Low => {
                        commands.spawn(BuyOrder {
                            goods_type: GoodsType::Food,
                            amount: 1,
                            price: food_price,
                            order_type: OrderType::Person,
                        });
                    },      
                    Income::Middle => {
                        commands.spawn(BuyOrder {
                            goods_type: GoodsType::Food,
                            amount: 2,
                            price: food_price,
                            order_type: OrderType::Person,
                        });
                    },  
                    Income::High => {
                        commands.spawn(BuyOrder {
                            goods_type: GoodsType::Food,
                            amount: 3,
                            price: food_price,
                            order_type: OrderType::Person,
                        });
                    },
                    Income::VeryHigh => {
                        commands.spawn(BuyOrder {
                            goods_type: GoodsType::Food,
                            amount: 4,  
                            price: food_price,
                            order_type: OrderType::Person,
                        });
                    },
                }
            },
        }
    }
}

fn market_loop(buy_query: Query<&BuyOrder>,
    buy_entities_query: Query<Entity, With<BuyOrder>>,
    sell_query: Query<&SellOrder>,
    mut commands: Commands) {
    let mut buy_orders: HashMap<GoodsType, Vec<BuyOrder>> = HashMap::new();
    let mut buy_order_sums: HashMap<GoodsType, BuyOrderSum> = HashMap::new();
    for buy_order in buy_query.iter() {
        buy_orders.entry(buy_order.goods_type)
            .or_insert_with(Vec::new)
            .push(buy_order.clone());
    }
    for orders in buy_orders {
        let buy_order_sum = BuyOrderSum {
            goods_type: orders.0,
            amount: 0,
            price: 0.0,
        };
        let sum = orders.1.iter().fold(buy_order_sum, |mut sum, order| {
            sum.amount += order.amount;
            sum.price = order.price;
            sum
        });
        buy_order_sums.insert(orders.0, sum);
    }
    buy_order_sums.iter().for_each(|(goods_type, buy_order_sum)| {
        println!("Goods Type: {:?}, Amount: {:?}, Price: {:?}", goods_type, buy_order_sum.amount, buy_order_sum.price);
    });

    for buy_order in buy_entities_query.iter() {
        commands.entity(buy_order).despawn();
    }
    // for sell_order in sell_query.iter() {
    // }
}
