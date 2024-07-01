use wgpu::util::DeviceExt;
use wgpu::{BufferDescriptor, BufferUsages, Device, Queue};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use futures_intrusive::channel::shared::oneshot_channel;

use crate::tokenizer::Tokenizer;

const MAX_WORKGROUP_SIZE: u32 = 65535;

pub fn prep_token_data(source: &str, vocab_size: usize) -> (IndexMap<String, usize>, Vec<Vec<usize>>) {
    let mut token_list = IndexMap::new();
    let mut token_indices: Vec<Vec<usize>> = Vec::new();
    token_list.insert("</w>".to_string(), 0);

    let token_indices: Vec<Vec<usize>> = source.split_whitespace()
        .map(|word| {
            word.chars()
                .map(|c| {
                    let s = c.to_string().to_lowercase();
                    let index = token_list.iter().position(|(key, _)| key == &s).unwrap_or_else(|| {
                        token_list.insert(s.clone(), token_list.len());
                        token_list.len() - 1
                    });
                    index
                })
                .chain(std::iter::once(0)) // Add index for "</w>"
                .collect::<Vec<_>>()
        })
        .collect();

    (token_list, token_indices)
}

pub async fn initialize_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();
    (device, queue)
}  

pub async fn run_gpu_pipeline(device: &wgpu::Device, queue: &wgpu::Queue, token_indices: &[u32], vocab_size: usize) -> HashMap<(usize, usize), usize> {
    let pair_frequencies_size = vocab_size * vocab_size;
    let mut pair_frequencies = vec![vec![0u32; vocab_size]; vocab_size];
    let mut output = HashMap::new();

    // Buffer initialization
    let token_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Token Buffer"),
        contents: bytemuck::cast_slice(&token_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let pair_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pair Frequencies Buffer"),
        size: (pair_frequencies_size * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Read Buffer"),
        size: (pair_frequencies_size * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Load compute shader
    let shader_source = include_str!("pair_freq_shader.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Bind Group Layout"),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: token_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pair_buffer.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    });

    // Create the pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create the compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
        compilation_options: Default::default()  
    });

    // Dispatch the compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let total_work_items = token_indices.len() as u32;
        let workgroup_size = 64u32;
        let max_dispatch_size = 65535u32;
        let total_dispatches = (total_work_items + workgroup_size - 1) / workgroup_size;

        for dispatch in (0..total_dispatches).step_by(max_dispatch_size as usize) {
            let dispatch_count = (total_dispatches - dispatch).min(max_dispatch_size);
            compute_pass.dispatch_workgroups(dispatch_count, 1, 1);
        }
    }

    // Copy data from pair_buffer to read_buffer
    encoder.copy_buffer_to_buffer(&pair_buffer, 0, &read_buffer, 0, (pair_frequencies_size * std::mem::size_of::<u32>()) as u64);
    queue.submit(Some(encoder.finish()));

    // Read back the data
    let buffer_slice = read_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| sender.send(result).unwrap());

    device.poll(wgpu::Maintain::Wait);
    let result = receiver.await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let gpu_result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buffer.unmap();

    for (i, &value) in gpu_result.iter().enumerate() {
        let row = i / vocab_size;
        let col = i % vocab_size;

        if row < vocab_size && col < vocab_size {
            pair_frequencies[row][col] += value;
        }
    }

    for row in 0..vocab_size {
        for col in 0..vocab_size {
            let frequency = pair_frequencies[row][col];
            if frequency > 0 {
                output.insert((row, col), frequency as usize);
            }
        }
    }

    output
}

pub async fn train_gpu(source: &str, iterations: usize, vocab_size: usize, output_filepath: Option<&str>) -> Self {
    // Train tokenizer on GPU using byte pair encoding
    let filepath = output_filepath.unwrap_or("./src/tokenizer.json");
    let mut merge_rules: Vec<(String, String)> = Vec::new();
    let mut token_list: Vec<String> = vec!["</w>".to_string()];
    let mut token_map: HashMap<String, u32> = HashMap::new();
    token_map.insert("</w>".to_string(), 0);

    // Initialize token indices
    let mut token_indices: Vec<u32> = source.split_whitespace()
        .flat_map(|word| {
            word.chars()
                .map(|c| {
                    let s = c.to_string().to_lowercase();
                    if let Some(&index) = token_map.get(&s) {
                        index
                    } else {
                        let index = token_list.len() as u32;
                        token_list.push(s.clone());
                        token_map.insert(s, index);
                        index
                    }
                })
                .chain(std::iter::once(0)) // Add index for "</w>"
                .collect::<Vec<u32>>()
        })
        .collect();

    let start_time = Instant::now();
    let (device, queue) = initialize_wgpu().await;

    for i in 0..iterations {
        let iter_time = Instant::now();
        let pair_frequencies = run_gpu_pipeline(&device, &queue, &token_indices, vocab_size).await;
        let max_pair = pair_frequencies.par_iter().max_by_key(|&(_, &count)| count);

        if let Some(((first, second), _)) = max_pair {
            let first = *first as u32;
            let second = *second as u32;
            let new_token = format!("{}{}", token_list[first as usize], token_list[second as usize]);
            let new_index = token_list.len() as u32;
            token_list.push(new_token.clone());
            merge_rules.push((token_list[first as usize].clone(), token_list[second as usize].clone()));

            let update_time = Instant::now();
            let mut new_token_indices = Vec::with_capacity(token_indices.len());
            let mut iter = token_indices.iter().peekable();
            while let Some(&current) = iter.next() {
                if iter.peek() == Some(&&second) && current == first {
                    new_token_indices.push(new_index);
                    iter.next(); // Skip the second token in the pair
                } else {
                    new_token_indices.push(current);
                }
            }
            token_indices = new_token_indices;

        } else {
            break; // No more pairs to merge
        }

        // Save tokenizer every 50 iterations
        if i % 50 == 0 {
            let vocabulary: HashSet<String> = token_list.clone().into_iter().collect();
            let tokenizer = Tokenizer::new(vocabulary, merge_rules.clone());
            tokenizer.save(&format!("{}", filepath)).unwrap();

            // Log progress and metrics
            let elapsed_time = start_time.elapsed().as_secs_f32();
            let percent = 100.0 * ((i + 1) as f32 / iterations as f32);
            let iteration_time = elapsed_time / (i as f32 + 1.0);
            print!("\rTraining Tokenizer: {:.2}% | ETA: {:.2}s", percent, iteration_time * (iterations as f32 - (i as f32 + 1.0)));
            io::stdout().flush().unwrap();
        }
        println!("Iteration {} time: {:?}", i, iter_time.elapsed().as_secs_f32());
    }

    let vocabulary: HashSet<String> = token_list.into_iter().collect();
    Tokenizer::new(vocabulary, merge_rules)
}
