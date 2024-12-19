#include <iostream>
#include <vector>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/target.hpp>

int main(void)
{
    migraphx::program prog;
    migraphx::module *main_module = prog.get_main_module();
    
    migraphx::shape shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<float> input2 = {17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
    
    migraphx::instruction_ref lit1 = main_module->add_literal(migraphx::literal{shape, input1});
    migraphx::instruction_ref lit2 = main_module->add_literal(migraphx::literal{shape, input2});
    
    main_module->add_instruction(migraphx::make_op("dot"), lit1, lit2);
    prog.compile(migraphx::gpu::target{});
    
    migraphx::argument result = prog.eval({}).back();
    
    std::vector<float> result_data(shape.elements());
    result.visit([&](auto output) { result_data.assign(output.begin(), output.end()); });
    
    for (size_t i = 0; i < result_data.size(); i++)
    {
        std::cout << result_data[i] << " ";
        
        if (!((i + 1) % 4))
        {
            std::cout << std::endl;
        }
    }

    return 0;
}
