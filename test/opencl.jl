using KernelAbstractions, OpenCL, pocl_jll
using SIMD
using Test

const backend = OpenCLBackend()

@testset "OpenCL SIMD Load/Store Tests" begin

    @testset "Basic load/store operations" begin
        @kernel function load_store_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vload(Vec{4, Float32}, b, i)
            @inbounds vstore(xs + Vec{4, Float32}(1f0), a, i)
        end

        b = KernelAbstractions.zeros(backend, Float32, 1024)
        a = similar(b)
        load_store_kernel!(backend)(a, b; ndrange = 1024 ÷ 4)
        @test all(==(1f0), a)
    end

    @testset "Multiple data types load/store" begin
        # Test Int32 vectors
        @kernel function load_store_int32_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vload(Vec{4, Int32}, b, i)
            @inbounds vstore(xs + Vec{4, Int32}(10), a, i)
        end

        b_int = KernelAbstractions.zeros(backend, Int32, 256)
        fill!(b_int, 5)
        a_int = similar(b_int)
        load_store_int32_kernel!(backend)(a_int, b_int; ndrange = 256 ÷ 4)
        @test all(==(15), a_int)

        # Test different vector sizes for Float32
        @kernel function load_store_float32_vec8_kernel!(a, b)
            i = 8 * (@index(Global) - 1) + 1
            xs = @inbounds vload(Vec{8, Float32}, b, i)
            @inbounds vstore(xs * Vec{8, Float32}(2f0), a, i)
        end

        b_f32 = KernelAbstractions.ones(backend, Float32, 512)
        a_f32 = similar(b_f32)
        load_store_float32_vec8_kernel!(backend)(a_f32, b_f32; ndrange = 512 ÷ 8)
        @test all(==(2f0), a_f32)
    end

    @testset "Aligned load/store operations" begin
        @kernel function aligned_load_store_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vloada(Vec{4, Float32}, b, i)
            @inbounds vstorea(xs + Vec{4, Float32}(3f0), a, i)
        end

        b = KernelAbstractions.zeros(backend, Float32, 1024)
        fill!(b, 2f0)
        a = similar(b)
        aligned_load_store_kernel!(backend)(a, b; ndrange = 1024 ÷ 4)
        @test all(==(5f0), a)
    end

    @testset "Non-temporal load/store operations" begin
        @kernel function nontemporal_load_store_kernel!(a, b)
            i = 4 * (@index(Global) - 1) + 1
            xs = @inbounds vloadnt(Vec{4, Float32}, b, i)
            @inbounds vstorent(xs * Vec{4, Float32}(1.5f0), a, i)
        end

        b = KernelAbstractions.ones(backend, Float32, 512)
        fill!(b, 4f0)
        a = similar(b)
        nontemporal_load_store_kernel!(backend)(a, b; ndrange = 512 ÷ 4)
        @test all(==(6f0), a)
    end

    @testset "Masked load/store operations" begin
        @kernel function masked_load_store_kernel!(a, b, masks_buf)
            idx = @index(Global)
            i = 4 * (idx - 1) + 1

            # Create mask from buffer (convert to bool)
            mask = Vec{4, Bool}((
                masks_buf[i] > 0,
                masks_buf[i+1] > 0,
                masks_buf[i+2] > 0,
                masks_buf[i+3] > 0
            ))

            xs = @inbounds vload(Vec{4, Float32}, b, i, mask)
            result = xs + Vec{4, Float32}(10f0)
            @inbounds vstore(result, a, i, mask)
        end

        n = 256
        b = KernelAbstractions.ones(backend, Float32, n)
        a = KernelAbstractions.zeros(backend, Float32, n)

        # Create mask pattern: alternating true/false
        mask_data = zeros(Int32, n)
        for i in 1:n
            mask_data[i] = i % 2
        end
        masks_buf = KernelAbstractions.zeros(backend, Int32, n)
        copyto!(masks_buf, mask_data)

        masked_load_store_kernel!(backend)(a, b, masks_buf; ndrange = n ÷ 4)

        a_host = Array(a)
        # Check that masked elements were updated, unmasked remain zero
        for i in 1:n
            if i % 2 == 1  # mask was true
                @test a_host[i] == 11f0
            else  # mask was false
                @test a_host[i] == 0f0
            end
        end
    end

    @testset "Gather/scatter operations" begin
        @kernel function gather_scatter_kernel!(a, b, indices_buf)
            idx = @index(Global)
            base_i = 4 * (idx - 1) + 1

            # Load indices
            indices = Vec{4, Int32}((
                indices_buf[base_i],
                indices_buf[base_i + 1],
                indices_buf[base_i + 2],
                indices_buf[base_i + 3]
            ))

            # Gather from b using indices
            gathered = @inbounds vgather(b, indices)

            # Process and scatter back to a
            result = gathered * Vec{4, Float32}(2f0)
            @inbounds vscatter(result, a, indices)
        end

        n = 128
        b = KernelAbstractions.zeros(backend, Float32, n)
        # Fill b with indices as values for easy verification
        b_data = Float32.(1:n)
        copyto!(b, b_data)

        a = KernelAbstractions.zeros(backend, Float32, n)

        # Create gather indices (1-based)
        indices_data = Int32[]
        for i in 1:4:(n-3)
            # Gather in reverse order for testing
            append!(indices_data, [i+3, i+2, i+1, i])
        end
        indices_buf = KernelAbstractions.zeros(backend, Int32, length(indices_data))
        copyto!(indices_buf, indices_data)

        gather_scatter_kernel!(backend)(a, b, indices_buf; ndrange = length(indices_data) ÷ 4)

        a_host = Array(a)
        # Verify scattered results
        for (i, original_idx) in enumerate(indices_data)
            expected = Float32(original_idx) * 2f0
            @test a_host[original_idx] == expected
        end
    end

    @testset "Masked gather/scatter operations" begin
        @kernel function masked_gather_scatter_kernel!(a, b, indices_buf, masks_buf)
            idx = @index(Global)
            base_i = 4 * (idx - 1) + 1

            # Load indices and mask
            indices = Vec{4, Int32}((
                indices_buf[base_i],
                indices_buf[base_i + 1],
                indices_buf[base_i + 2],
                indices_buf[base_i + 3]
            ))

            mask = Vec{4, Bool}((
                masks_buf[base_i] > 0,
                masks_buf[base_i + 1] > 0,
                masks_buf[base_i + 2] > 0,
                masks_buf[base_i + 3] > 0
            ))

            # Masked gather
            gathered = @inbounds vgather(b, indices, mask)

            # Process and masked scatter
            result = gathered + Vec{4, Float32}(100f0)
            @inbounds vscatter(result, a, indices, mask)
        end

        n = 64
        b = KernelAbstractions.zeros(backend, Float32, n)
        b_data = Float32.(1:n) .* 10f0
        copyto!(b, b_data)

        a = KernelAbstractions.zeros(backend, Float32, n)

        # Create indices and masks
        indices_data = Int32.(1:n)
        indices_buf = KernelAbstractions.zeros(backend, Int32, n)
        copyto!(indices_buf, indices_data)

        # Checkerboard mask pattern
        mask_data = [i % 2 for i in 1:n]
        masks_buf = KernelAbstractions.zeros(backend, Int32, n)
        copyto!(masks_buf, mask_data)

        masked_gather_scatter_kernel!(backend)(a, b, indices_buf, masks_buf; ndrange = n ÷ 4)

        a_host = Array(a)
        for i in 1:n
            if i % 2 == 1  # mask was true
                expected = Float32(i) * 10f0 + 100f0
                @test a_host[i] == expected
            else  # mask was false, should remain zero
                @test a_host[i] == 0f0
            end
        end
    end

    if Base.libllvm_version >= v"9" || Sys.CPU_NAME == "skylake"
        @testset "Expand load operations" begin
            @kernel function expandload_kernel!(a, b, masks_buf)
                idx = @index(Global)
                base_i = 4 * (idx - 1) + 1

                mask = Vec{4, Bool}((
                    masks_buf[base_i] > 0,
                    masks_buf[base_i + 1] > 0,
                    masks_buf[base_i + 2] > 0,
                    masks_buf[base_i + 3] > 0
                ))

                # Expand load - loads compressed data based on mask
                expanded = @inbounds vloadx(b, base_i, mask)
                @inbounds vstore(expanded, a, base_i)
            end

            n = 64
            b = KernelAbstractions.zeros(backend, Float32, n)
            b_data = Float32.(1:n)
            copyto!(b, b_data)

            a = KernelAbstractions.zeros(backend, Float32, n)

            # Test with specific mask pattern
            mask_data = zeros(Int32, n)
            for i in 1:4:n
                mask_data[i:min(i+3,n)] = [1, 0, 0, 1]  # Load 1st and 4th elements
            end
            masks_buf = KernelAbstractions.zeros(backend, Int32, n)
            copyto!(masks_buf, mask_data)

            expandload_kernel!(backend)(a, b, masks_buf; ndrange = n ÷ 4)

            a_host = Array(a)
            # Verify expand load results
            for i in 1:4:n
                # Should have loaded elements i and i+3, others should be 0
                @test a_host[i] == Float32(i)      # 1st element loaded
                @test a_host[i+1] == 0f0           # 2nd element masked (zero)
                @test a_host[i+2] == 0f0           # 3rd element masked (zero)
                if i+3 <= n
                    @test a_host[i+3] == Float32(i+1)  # 4th element loaded (from next position in source)
                end
            end
        end

        @testset "Compress store operations" begin
            @kernel function compressstore_kernel!(a, b, masks_buf)
                idx = @index(Global)
                base_i = 4 * (idx - 1) + 1

                data = @inbounds vload(Vec{4, Float32}, b, base_i)

                mask = Vec{4, Bool}((
                    masks_buf[base_i] > 0,
                    masks_buf[base_i + 1] > 0,
                    masks_buf[base_i + 2] > 0,
                    masks_buf[base_i + 3] > 0
                ))

                # Compress store - stores only masked elements consecutively
                @inbounds vstorec(data, a, base_i, mask)
            end

            n = 64
            b = KernelAbstractions.zeros(backend, Float32, n)
            b_data = Float32.(1:n) .* 5f0
            copyto!(b, b_data)

            a = KernelAbstractions.zeros(backend, Float32, n)

            # Test with specific mask pattern
            mask_data = zeros(Int32, n)
            for i in 1:4:n
                mask_data[i:min(i+3,n)] = [1, 0, 1, 0]  # Store 1st and 3rd elements
            end
            masks_buf = KernelAbstractions.zeros(backend, Int32, n)
            copyto!(masks_buf, mask_data)

            compressstore_kernel!(backend)(a, b, masks_buf; ndrange = n ÷ 4)

            a_host = Array(a)
            # Verify compress store results - masked elements stored consecutively
            for group in 1:4:(n-3)
                # Each group should have 2 elements stored consecutively
                @test a_host[group] == Float32(group) * 5f0      # 1st element
                @test a_host[group+1] == Float32(group+2) * 5f0  # 3rd element (compressed)
                # Elements beyond the compressed ones should remain zero
                @test a_host[group+2] == 0f0
                @test a_host[group+3] == 0f0
            end
        end
    else
        @info "Skipping expandload/compressstore tests" Base.libllvm_version Sys.CPU_NAME
    end

end
