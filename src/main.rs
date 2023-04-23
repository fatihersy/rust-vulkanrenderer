use anyhow::{Context, Result};
use ash::{
    self,
    vk::{self, DeviceQueueCreateInfo},
};
use gpu_allocator::vulkan::*;

fn main() -> Result<()> {
    // CONFIG
    let width: u64 = 720;
    let height: u64 = 720;
    let value_count = width * height;
    let value = 255 << 24;

    // CONTEXT
    let entry = unsafe { ash::Entry::load() }?;

    let instance = {
        let application_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);
        let create_info = vk::InstanceCreateInfo::builder().application_info(&application_info);
        unsafe { entry.create_instance(&create_info, None) }?
    };

    let physical_device = unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .next()
        .context("No physical device found!")?;

    let device = {
        let queue_priorities = [1.0];
        let queue_create_info = DeviceQueueCreateInfo::builder()
            .queue_family_index(0)
            .queue_priorities(&queue_priorities);
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        unsafe { instance.create_device(physical_device, &create_info, None) }?
    };

    let queue = unsafe { device.get_device_queue(0, 0) };

    // CREATING ALLOCATOR
    let mut allocator = {
        let allocator_create_desc = AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        Allocator::new(&allocator_create_desc)?
    };

    let buffer = {
        let create_info = vk::BufferCreateInfo::builder()
            .size(value_count * std::mem::size_of::<i32>() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        unsafe { device.create_buffer(&create_info, None) }.unwrap()
    };

    let allocation = {
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_create_description = AllocationCreateDesc {
            name: "Buffer Allocation",
            requirements: memory_requirements,
            location: gpu_allocator::MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = allocator.allocate(&allocation_create_description)?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
            .unwrap();
        allocation
    };

    let command_pool = {
        let create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(0);
        unsafe { device.create_command_pool(&create_info, None) }?
    };

    let command_buffer = {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);
        unsafe { device.allocate_command_buffers(&create_info) }?
            .into_iter()
            .next()
            .context("Allocation of Command Buffer failed!")?
    };

    // RECORDING COMMAND BUFFER
    {
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }?
    }

    unsafe {
        device.cmd_fill_buffer(
            command_buffer,
            buffer,
            allocation.offset(),
            allocation.size(),
            value,
        )
    }

    unsafe { device.end_command_buffer(command_buffer) }?;

    let fence = {
        let create_info = vk::FenceCreateInfo::builder().build();
        unsafe { device.create_fence(&create_info, None) }?
    };

    // EXECUTE THE COMMAND BUFFER BY UPLOADING IT TO THE GPU THROUGH THE QUEUE
    {
        let submit_info =
            vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            device
                .queue_submit(queue, std::slice::from_ref(&submit_info), fence)
                .unwrap()
        };
    };

    // WAIT FOR THE EXECUTION TO COMPLETE
    unsafe {
        device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
    }

    let data = allocation
        .mapped_slice()
        .context("Cannot Able to Access Buffer from Host")?;

    // READ THE DATA
    image::save_buffer(
        "image.png",
        data,
        width as u32,
        height as u32,
        image::ColorType::Rgba8,
    )?;

    // CLEAN UP
    unsafe { device.destroy_fence(fence, None) }
    unsafe { device.destroy_command_pool(command_pool, None) }

    allocator.free(allocation).unwrap();
    drop(allocator);
    unsafe { device.destroy_buffer(buffer, None) }

    unsafe { device.destroy_device(None) }
    unsafe { instance.destroy_instance(None) }
    Ok(())
}
